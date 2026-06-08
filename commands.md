# Assignment Commands

---

## #1 Matrix Transpose

```cmd
cd assigments\matrix-transpose

rem Compile
nvcc -O2 -o matrix-transpose.exe matrix-transpose.cu

rem Run
matrix-transpose.exe

rem Profile
nsys profile --stats=true -o transpose-report .\matrix-transpose.exe
```

---

## #2 Convolution Filter

```cmd
cd assigments\convolution-filter

rem Compile
nvcc -O2 -o convolution_filter.exe convolution_filter.cu

rem Run
convolution_filter.exe

rem Profile
nsys profile --stats=true -o convolution-report .\convolution_filter.exe
```

---

## #3 HDR Tone Mapping  (CMake + OpenCV)

```cmd
cd assigments\hdr-tone-mapping\cuda-webcam-filter

rem Configure
mkdir build && cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="C:\opencv\opencv\build\x64\vc16\lib" -DCMAKE_POLICY_VERSION_MINIMUM=3.5

rem Build
cmake --build .

rem Add OpenCV to PATH (required every new terminal session)
set PATH=C:\opencv\opencv\build\x64\vc16\bin;%PATH%

rem Run — global Reinhard (webcam)
cuda-webcam-filter.exe -f hdr --hdr-algo reinhard --exposure 2.5 --preview

rem Run — ACES filmic (synthetic gradient)
cuda-webcam-filter.exe -f hdr --hdr-algo aces -i synthetic -s gradient --exposure 3.0 --preview

rem Run — local Reinhard (synthetic checkerboard)
cuda-webcam-filter.exe -f hdr --hdr-algo local -i synthetic -s checkerboard --exposure 2.5 --gamma 2.2 --saturation 1.2 --preview

rem Profile
nsys profile --stats=true -o hdr-report .\cuda-webcam-filter.exe -f hdr --hdr-algo reinhard -i synthetic -s gradient
```

---

## #4 Filter Pipeline

### A. Standalone benchmark (no OpenCV)

```cmd
cd assigments\filter-pipeline

rem Compile
nvcc -O2 -arch=native -o filter_pipeline.exe filter_pipeline.cu

rem Run full sweep -> pipeline_benchmark.csv
filter_pipeline.exe

rem Quick smoke test
filter_pipeline.exe --quick

rem Plot results (requires matplotlib)
python -m pip install matplotlib pillow
python plot_benchmarks.py

rem Profile
nsys profile --stats=true -o pipeline-report .\filter_pipeline.exe --quick
```

### B. Integrated OpenCV app

```cmd
cd assigments\filter-pipeline\cuda-webcam-filter

rem Configure
mkdir build && cd build
cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="C:\opencv\opencv\build\x64\vc16\lib" -DCMAKE_POLICY_VERSION_MINIMUM=3.5

rem Build
cmake --build .

rem Add OpenCV to PATH
set PATH=C:\opencv\opencv\build\x64\vc16\bin;%PATH%

rem --- input sources ---

rem Webcam (default)
cuda-webcam-filter.exe --pipeline "blur:5,sharpen,edge" --preview

rem Static image
cuda-webcam-filter.exe --pipeline "blur:5,sharpen,edge" -i image -p C:\path\to\photo.jpg --preview

rem Video file
cuda-webcam-filter.exe --pipeline "blur:5,sharpen,edge" -i video -p C:\path\to\clip.mp4 --preview

rem Synthetic gradient
cuda-webcam-filter.exe --pipeline "blur:5,sharpen,edge" -i synthetic -s gradient --preview

rem Synthetic checkerboard
cuda-webcam-filter.exe --pipeline "emboss,grayscale" -i synthetic -s checkerboard --preview

rem --- multi-stream ---

rem Multi-stream across 4 bands
cuda-webcam-filter.exe --pipeline "blur:5,sharpen,edge" --multi-stream --streams 4 --preview

rem --- wipe transition ---

rem Wipe transition (synthetic input, no webcam needed)
cuda-webcam-filter.exe --pipeline "blur:5,sharpen,edge" --transition "sepia,emboss" --transition-duration 2.0 --wipe-softness 60 -i synthetic -s gradient --preview
```
