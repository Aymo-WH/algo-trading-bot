from runpod_flash import Endpoint, GpuGroup
import asyncio

@Endpoint(name="flash-quickstart", gpu=GpuGroup.ADA_24, dependencies=["torch"], workers=(1, 3))
def gpu_compute(data):
  import torch
  tensor = torch.tensor(data, device="cuda")
  return {"result": tensor.sum().item(), "device": torch.cuda.get_device_name(0)}

async def main():
  result = await gpu_compute([1, 2, 3, 4, 5])
  print(f"Sum: {result['result']}\nComputed on: {result['device']}")

if __name__ == "__main__":
  asyncio.run(main())