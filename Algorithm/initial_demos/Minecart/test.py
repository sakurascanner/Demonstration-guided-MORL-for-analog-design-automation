import numpy as np

file_path = "/home/tanjunfeng/MORL/DG-MORL-main/Algorithm/initial_demos/Minecart/demos.npy"
data = np.load(file_path, allow_pickle=True)

print("type:", type(data))

if hasattr(data, 'shape'):
    print("shape:", data.shape)

print("\n content:")
print(data)

print("\n some content:")
for i in range(min(2, len(data))):
    print(f"\n--- trajectory {i} ---")
    if isinstance(data[i], dict):
        for key, value in data[i].items():
            print(f"  {key}: shape={getattr(value, 'shape', 'N/A')}, type={type(value)}, len={len(value) if hasattr(value, '__len__') else 'N/A'}")
    else:
        print(f"  the data: {data[i]}")