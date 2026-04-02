import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

import torch

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONUTF8'] = '1'

# Set random seeds
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Import the compiled extension
try:
    import triple_conv
    print("Successfully imported triple_conv extension")
except ImportError as e:
    print(f"Import failed: {e}")
    # If direct import fails, try dynamic loading
    from torch.utils.cpp_extension import load
    triple_conv = load(
        name="triple_conv",
        sources=["triple_conv.cpp", "triple_conv_kernel.cu"],
        verbose=True
    )

class TripleConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight1, weight3, weight5):
        # Save inputs for backward propagation
        ctx.save_for_backward(input, weight1, weight3, weight5)
        # Call C++/CUDA forward propagation
        return triple_conv.triple_conv(input, weight1, weight3, weight5)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved inputs
        input, weight1, weight3, weight5 = ctx.saved_tensors
        # Call C++/CUDA backward propagation
        grads = triple_conv.backward(grad_output, input, weight1, weight3, weight5)
        # Return gradients: input_grad, weight1_grad, weight3_grad, weight5_grad
        return grads[0], grads[1], grads[2], grads[3]

class TripleConvLayer(nn.Module):
    """
    Triple convolution fusion layer
    Combines 1x1, 3x3, 5x5 convolution kernels through a single convolution operation
    """
    def __init__(self, in_channels, out_channels):
        super(TripleConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Use extension function to create weights
        weights = triple_conv.create_weights(in_channels, out_channels)
        self.weight1 = nn.Parameter(weights[0])  # 1x1 convolution weights
        self.weight3 = nn.Parameter(weights[1])  # 3x3 convolution weights  
        self.weight5 = nn.Parameter(weights[2])  # 5x5 convolution weights
    
    def forward(self, x):
        """
        Forward propagation
        Args:
            x: Input tensor [batch, in_channels, height, width]
        Returns:
            output: Output tensor [batch, out_channels, height, width]
        """
        return TripleConvFunction.apply(x, self.weight1, self.weight3, self.weight5)
    
    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}'

def test_basic_functionality():
    """Test basic functionality"""
    print("=== Testing Basic Functionality ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data
    batch_size = 2
    in_channels = 3
    out_channels = 8
    height, width = 16, 16
    
    # Create input
    x = torch.randn(batch_size, in_channels, height, width, device=device, requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Create custom layer
    layer = TripleConvLayer(in_channels, out_channels).to(device)


    from fvcore.nn.parameter_count import parameter_count_table
    from fvcore.nn.flop_count import flop_count
    from fvcore.nn import FlopCountAnalysis
    print('end111')
    print(parameter_count_table(layer))

    flops = FlopCountAnalysis(layer, x )
    print(f"Total FLOPs: {flops.total()}")

    
    # Forward propagation
    output = layer(x)
    print(f"Output shape: {output.shape}")
    
    # Backward propagation
    loss = output.sum()
    loss.backward()
    
    print("Gradient calculation completed")
    print(f"weight1 gradient: {layer.weight1.grad is not None}")
    print(f"weight3 gradient: {layer.weight3.grad is not None}") 
    print(f"weight5 gradient: {layer.weight5.grad is not None}")
    print(f"Input gradient: {x.grad is not None}")




def test_performance_comparison():
    """Performance comparison test"""
    print("\n=== Performance Comparison Test ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameter settings
    batch_size = 4
    in_channels = 32
    out_channels = 64
    height, width = 32, 32
    
    # Create custom layer
    custom_layer = TripleConvLayer(in_channels, out_channels).to(device)
    
    # Create standard PyTorch convolution layers for comparison
    conv_layers = nn.ModuleList([
        nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False).to(device),
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False).to(device), 
        nn.Conv2d(in_channels, out_channels, 5, padding=2, bias=False).to(device)
    ])
    
    # Use the same weight initialization for fair comparison
    with torch.no_grad():
        # Initialize all weights to the same small values to avoid gradient explosion
        init_weight = torch.randn(out_channels, in_channels, 1, 1, device=device)    # * 0.01
        custom_layer.weight1.data = init_weight.view(out_channels, -1).clone()
        conv_layers[0].weight.data = init_weight.clone()
        
        init_weight = torch.randn(out_channels, in_channels, 3, 3, device=device)    # * 0.01
        custom_layer.weight3.data = init_weight.view(out_channels, -1).clone()
        conv_layers[1].weight.data = init_weight.clone()
        
        init_weight = torch.randn(out_channels, in_channels, 5, 5, device=device)    # * 0.01
        custom_layer.weight5.data = init_weight.view(out_channels, -1).clone()
        conv_layers[2].weight.data = init_weight.clone()
    
    # Create identical input and ground truth
    x_shared = torch.randn(batch_size, in_channels, height, width, device=device)
    gt = torch.randn(batch_size, out_channels, height, width, device=device)
    
    # Custom layer performance test
    print("Testing custom layer...")
    
    # Create input with gradients for custom layer
    x1 = x_shared.clone().requires_grad_(True)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    try:
        custom_output = custom_layer(x1)
        custom_loss = (custom_output - gt).pow(2).mean()  # MSE loss
        custom_loss.backward()
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        custom_time = time.time() - start_time
        
        # Save weight gradients
        custom_weight_grads = []
        for param in custom_layer.parameters():
            if param.grad is not None:
                custom_weight_grads.append(param.grad.clone())
            else:
                custom_weight_grads.append(torch.zeros_like(param))
        
        # Save input gradient
        custom_input_grad = x1.grad.clone() if x1.grad is not None else torch.zeros_like(x1)
                
    except Exception as e:
        print(f"Custom layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Standard convolution layer performance test
    print("Testing standard convolution layers...")
    
    # Create input with gradients for separated convolutions
    x2 = x_shared.clone().requires_grad_(True)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    try:
        outputs = [conv(x2) for conv in conv_layers]
        separate_output = sum(outputs)
        separate_loss = (separate_output - gt).pow(2).mean()
        separate_loss.backward()
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        separate_time = time.time() - start_time

        # Extract weight gradients
        separate_weight_grads = []
        for conv in conv_layers:
            if conv.weight.grad is not None:
                separate_weight_grads.append(conv.weight.grad.clone())
            else:
                separate_weight_grads.append(torch.zeros_like(conv.weight))
        
        # Save input gradient
        separate_input_grad = x2.grad.clone() if x2.grad is not None else torch.zeros_like(x2)
                
    except Exception as e:
        print(f"Separated convolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Output comparison
    print("\n=== Output Comparison ===")
    print(f"Custom layer output range: [{custom_output.min().item():.6f}, {custom_output.max().item():.6f}]")
    print(f"Separated convolution output range: [{separate_output.min().item():.6f}, {separate_output.max().item():.6f}]")

    output_diff = (custom_output - separate_output).abs().max().item()
    print(f"Maximum output difference: {output_diff:.8f}")
    
    # Output statistical properties
    custom_mean, custom_std = custom_output.mean().item(), custom_output.std().item()
    separate_mean, separate_std = separate_output.mean().item(), separate_output.std().item()
    print(f"Custom layer output - mean: {custom_mean:.6f}, std: {custom_std:.6f}")
    print(f"Separated convolution output - mean: {separate_mean:.6f}, std: {separate_std:.6f}")

    # Weight gradient comparison
    print("\n=== Weight Gradient Comparison ===")
    print("Custom layer gradient norms:")
    for i, grad in enumerate(custom_weight_grads):
        print(f"  Weight{i+1}: {grad.norm().item():.6f}")
    
    print("Standard convolution layer gradient norms:")
    for i, grad in enumerate(separate_weight_grads):
        print(f"  Convolution layer{i+1}: {grad.norm().item():.6f}")

    # Check if weight gradients are similar
    if len(custom_weight_grads) == len(separate_weight_grads):
        for i in range(len(custom_weight_grads)):
            # Adjust shape to match
            if i < len(separate_weight_grads):
                separate_grad_reshaped = separate_weight_grads[i].view(out_channels, -1)
            else:
                print(f"Warning: Gradient index {i} out of range")
                continue
                
            if custom_weight_grads[i].shape == separate_grad_reshaped.shape:
                # Calculate absolute and relative differences
                grad_diff = (custom_weight_grads[i] - separate_grad_reshaped).abs().max().item()
                grad_norm = custom_weight_grads[i].norm().item()
                rel_diff = grad_diff / (grad_norm + 1e-8)  # Avoid division by zero
                
                print(f"Gradient{i+1} - absolute difference: {grad_diff:.8f}, relative difference: {rel_diff:.8f}")
                
                # Compare gradient norm ratios
                custom_norm = custom_weight_grads[i].norm().item()
                separate_norm = separate_grad_reshaped.norm().item()
                if custom_norm > 0 and separate_norm > 0:
                    ratio = max(custom_norm, separate_norm) / min(custom_norm, separate_norm)
                    print(f"Gradient{i+1} norm ratio: {ratio:.4f}")
                else:
                    print(f"Gradient{i+1} norm too small to compare")
            else:
                print(f"Gradient{i+1} shape mismatch: {custom_weight_grads[i].shape} vs {separate_grad_reshaped.shape}")
    else:
        print("Gradient count mismatch")

    # Input gradient comparison
    print("\n=== Input Gradient Comparison ===")
    print(f"Custom layer input gradient norm: {custom_input_grad.norm().item():.6f}")
    print(f"Separated convolution input gradient norm: {separate_input_grad.norm().item():.6f}")
    
    # Calculate input gradient differences
    input_grad_diff = (custom_input_grad - separate_input_grad).abs().max().item()
    input_grad_norm = custom_input_grad.norm().item()
    input_grad_rel_diff = input_grad_diff / (input_grad_norm + 1e-8)
    print(f"Input gradient - absolute difference: {input_grad_diff:.8f}, relative difference: {input_grad_rel_diff:.8f}")
    
    # Input gradient statistics
    print(f"Custom layer input gradient range: [{custom_input_grad.min().item():.6f}, {custom_input_grad.max().item():.6f}]")
    print(f"Separated convolution input gradient range: [{separate_input_grad.min().item():.6f}, {separate_input_grad.max().item():.6f}]")

    # Performance statistics
    print("\n=== Performance Statistics ===")
    speedup = separate_time / custom_time if custom_time > 0 else float('inf')
    print(f"Custom layer time: {custom_time:.6f}s")
    print(f"Separated convolution time: {separate_time:.6f}s") 
    print(f"Speedup: {speedup:.2f}x")
    
    # Return results for further analysis
    return {
        'output_diff': output_diff,
        'weight_grad_diffs': [grad_diff],
        'input_grad_diff': input_grad_diff,
        'custom_time': custom_time,
        'separate_time': separate_time,
        'custom_weight_grads': custom_weight_grads,
        'separate_weight_grads': separate_weight_grads,
        'custom_input_grad': custom_input_grad,
        'separate_input_grad': separate_input_grad
    }




def test_integration():
    """Integration test - Use in a simple network"""
    print("\n=== Integration Test ===")
    
    class SimpleNet(nn.Module):
        def __init__(self, in_channels=3, num_classes=10):
            super(SimpleNet, self).__init__()
            self.conv1 = TripleConvLayer(in_channels, 32)
            self.conv2 = TripleConvLayer(32, 64) 
            self.conv3 = TripleConvLayer(64, 128)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, num_classes)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x) 
            x = self.conv3(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNet().to(device)
    
    # Test forward propagation
    x = torch.randn(4, 3, 32, 32, device=device)
    output = model(x)
    print(f"Network output shape: {output.shape}")
    
    # Test backward propagation
    loss = output.sum()
    loss.backward()
    print("Network backward propagation completed")
    
    # Check all parameters have gradients
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"Warning: {name} has no gradient")
        else:
            print(f"{name}: gradient norm = {param.grad.norm().item():.6f}")

if __name__ == "__main__":
    # Run all tests
    # test_basic_functionality()
    test_performance_comparison() 
    # test_integration()
    print("\n=== All tests completed ===")