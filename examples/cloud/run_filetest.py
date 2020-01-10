import os

print(os.getcwd())
print(os.listdir("/gcp_output"))
with open("/gcp_output/filetest.txt", "w+") as f:
    f.write("test")
