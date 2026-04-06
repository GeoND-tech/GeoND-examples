import torch
import geondpt as gpt
import geondptfree as gptfree
import time

pb = gpt.Paraboloid(512, 1000).cuda()
pbfree = gptfree.Paraboloid(512, 1000).cuda()
pbinput = torch.rand(256, 512).cuda()
pboutput = torch.rand(256, 1000).cuda()
criterion = torch.nn.MSELoss()

for i in range(1, 100): #warmup
  tmp = pb(pbinput)
  tmp = pbfree(pbinput)

print("\nPARABOLOID\n")

torch.cuda.synchronize()
start = time.perf_counter()
for i in range(1, 1000):
  tmp = pb(pbinput)

torch.cuda.synchronize()
finish = time.perf_counter()
elapsed = finish - start
print(f"Paraboloid forward, 1000 runs: {elapsed:.5f} seconds")



torch.cuda.synchronize()
start = time.perf_counter()
for i in range(1, 1000):
  tmp = pbfree(pbinput)

torch.cuda.synchronize()
finish = time.perf_counter()
elapsed = finish - start
print(f"Paraboloid forward (free), 1000 runs: {elapsed:.5f} seconds")

print("")

torch.cuda.synchronize()
start = time.perf_counter()
for i in range(1, 1000):
  tmp = pb(pbinput)
  loss = criterion(tmp,pboutput)
  loss.backward(retain_graph=True)
  pb.zero_grad()

torch.cuda.synchronize()
finish = time.perf_counter()
elapsed = finish - start
print(f"Paraboloid forward+backward, 1000 runs: {elapsed:.5f} seconds")



torch.cuda.synchronize()
start = time.perf_counter()
for i in range(1, 1000):
  tmp = pbfree(pbinput)
  loss = criterion(tmp,pboutput)
  loss.backward(retain_graph=True)
  pb.zero_grad()

torch.cuda.synchronize()
finish = time.perf_counter()
elapsed = finish - start
print(f"Paraboloid forward+backward (free), 1000 runs: {elapsed:.5f} seconds")



pbc = gpt.ParaConv2d(3, 16, kernel_size=3, stride=1, padding=1).cuda()
pbcfree = gptfree.ParaConv2d(3, 16, kernel_size=3, stride=1, padding=1).cuda()
pbcinput = torch.rand(64, 3, 32, 32).cuda()
pbcoutput = torch.rand(64, 16, 32, 32).cuda()

for i in range(1, 100): #warmup
  tmp = pbc(pbcinput)
  tmp = pbcfree(pbcinput)

print("\nPARACONV2D\n")

torch.cuda.synchronize()
start = time.perf_counter()
for i in range(1,1000):
  tmp = pbc(pbcinput)

torch.cuda.synchronize()
finish = time.perf_counter()
elapsed = finish - start
print(f"ParaConv2d forward, 1000 runs: {elapsed:.5f} seconds")



torch.cuda.synchronize()
start = time.perf_counter()
for i in range(1,1000):
  tmp = pbcfree(pbcinput)

torch.cuda.synchronize()
finish = time.perf_counter()
elapsed = finish - start
print(f"ParaConv2d forward (free), 1000 runs: {elapsed:.5f} seconds")

print("")

torch.cuda.synchronize()
start = time.perf_counter()
for i in range(1, 1000):
  tmp = pbc(pbcinput)
  loss = criterion(tmp,pbcoutput)
  loss.backward(retain_graph=True)
  pbc.zero_grad()

torch.cuda.synchronize()
finish = time.perf_counter()
elapsed = finish - start
print(f"ParaConv2d forward+backward, 1000 runs: {elapsed:.5f} seconds")



torch.cuda.synchronize()
start = time.perf_counter()
for i in range(1, 1000):
  tmp = pbcfree(pbcinput)
  loss = criterion(tmp,pbcoutput)
  loss.backward(retain_graph=True)
  pbcfree.zero_grad()

torch.cuda.synchronize()
finish = time.perf_counter()
elapsed = finish - start
print(f"ParaConv2d forward+backward (free), 1000 runs: {elapsed:.5f} seconds")

print("")
