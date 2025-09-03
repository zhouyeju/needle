from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        print(f"Creating CMake extension for {name} in {self.sourcedir}")

class BuildCUDAOperations(build_ext):
    def build_extension(self, ext: CMakeExtension):
        build_dir = os.path.join(ext.sourcedir, "build")
        source_dir = ext.sourcedir
        subprocess.run(["cmake", f"-B{build_dir}", f"-S{source_dir}"])
        subprocess.run(["cmake", "--build", build_dir])


setup(
    name="needle",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "needle": ["*.so"]
    },
    ext_modules=[CMakeExtension("ops")],
    cmdclass={"build_ext": BuildCUDAOperations},
    install_requires=[
        "pybind11>=2.6.0",
        "numpy",
        "transformers",
        "ml_dtypes"
    ]
)