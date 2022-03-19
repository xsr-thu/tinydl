from setuptools import setup
import os
import shutil


curdir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(curdir, "build")
libpath = os.path.join(build_dir, "_tinydl.cpython-38-x86_64-linux-gnu.so")

if not os.path.exists(libpath):
    print("Please build first")
    exit(0)
else:
    shutil.copy(libpath, "tinydl")   

setup(
    name="tinydl",
    version="0.1.0",
    install_requires=["numpy"],
    packages=["tinydl"],
    package_data={
        "tinydl": ["*.so"]
    },
)
