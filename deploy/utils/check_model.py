import torch

# Load model with JIT
model = torch.jit.load("/home/felixl/Downloads/booster_extract/booster/Gait/bin/lib/T1/t1_fdr.pt", map_location="cpu")
print("Model loaded successfully with TorchScript JIT")
print(f"Model type: {type(model)}")

# Get model graph information
print("\n=== Model Graph ===")
print(model.graph)

# Explore model structure
print("\n=== Model Structure ===")
print("Model named modules:")
try:
    for name, module in model.named_modules():
        print(f"  {name}: {type(module)}")
except:
    print("Could not enumerate named modules")

print("\nModel named parameters:")
try:
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
except:
    print("Could not enumerate named parameters")

# Based on the parameters, determine the correct input size
print("\n=== Model Architecture Analysis ===")
print("From parameter shapes:")
print("- obs_encoder input size: 104 (from obs_encoder.0.weight: [256, 104])")
print("- obs_encoder output size: 128 (from obs_encoder.4.weight: [128, 128])")
print("- actor output size: 23 actions (from actor.2.weight: [23, 128])")
print("- critic output size: 1 value (from critic.6.weight: [1, 128])")
print("- priv_decoder output size: 14 features (from priv_decoder.4.weight: [14, 128])")
print(f"- obs_ndim: {model.obs_ndim}")

# Test with correct input size
print("\n=== Testing with correct input shape ===")
correct_input_size = 104  # Based on obs_encoder.0.weight shape
test_input = torch.randn(1, correct_input_size)

print(f"Testing with input shape: {test_input.shape}")

model.eval()
try:
    with torch.no_grad():
        # Test the full model (actor only - as seen in graph)
        actor_output = model(test_input)
        print(f"✓ SUCCESS!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Actor output shape: {actor_output.shape}")
        
        # Test individual components
        print(f"\n=== Individual Component Testing ===")
        
        # Test obs_encoder
        obs_flat = test_input.flatten(-model.obs_ndim, -1)
        obs_emb = model.obs_encoder(obs_flat)
        print(f"obs_encoder: {obs_flat.shape} -> {obs_emb.shape}")
        
        # Test actor
        actor_out = model.actor(obs_emb)
        print(f"actor: {obs_emb.shape} -> {actor_out.shape}")
        
        # Test critic (if accessible)
        try:
            # For critic, we need to concatenate with privileged info
            # Let's try different input sizes for critic
            critic_input_size = 118  # From critic.0.weight: [256, 118]
            critic_test_input = torch.randn(1, critic_input_size)
            critic_output = model.critic(critic_test_input)
            print(f"critic: {critic_test_input.shape} -> {critic_output.shape}")
        except Exception as e:
            print(f"Could not test critic directly: {e}")
        
        # Test priv_decoder
        try:
            priv_output = model.priv_decoder(obs_emb)
            print(f"priv_decoder: {obs_emb.shape} -> {priv_output.shape}")
        except Exception as e:
            print(f"Could not test priv_decoder: {e}")
            
except Exception as e:
    print(f"✗ Failed: {e}")

print(f"\n=== SUMMARY ===")
print(f"Model Components:")
print(f"1. obs_encoder: [?, 104] -> [?, 128] (observation encoder)")
print(f"2. actor: [?, 128] -> [?, 23] (policy network - outputs actions)")
print(f"3. critic: [?, 118] -> [?, 1] (value network - needs obs + privileged info)")
print(f"4. priv_decoder: [?, 128] -> [?, 14] (privileged information decoder)")
print(f"5. logstd: [1, 23] (action noise standard deviation)")
print(f"\nMain model input: [batch_size, 104] -> [batch_size, 23] (actor output)")
print(f"This appears to be a robotics control model with 23 actuators/joints.")