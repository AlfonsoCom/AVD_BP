import os

command = "..\..\.\CarlaUE4.exe /Game/Maps/Town01 -windowed -carla-server -benchmark -fps=30 -ResX=8 -ResY=8"
os.system(f'cmd /k "{command}"')