//
// Created by root on 7/17/22.
//
// Device code
__global__ void MyKernel1(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}
// Device code
__global__ void MyKernel2(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char* devPtr = static_cast<char*>(devPitchedPtr.ptr);
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    for (int z = 0; z < depth; ++z) {
        char* slice = devPtr + z * slicePitch;
        for (int y = 0; y < height; ++y) {
            float* row = (float*)(slice + y * pitch);
            for (int x = 0; x < width; ++x) {
                float element = row[x];
            }
        }
    }
}
void test1() {
    // Host code
    int width = 64, height = 64;
    float* devPtr;
    size_t pitch;
    cudaMallocPitch(&devPtr, &pitch,width * sizeof(float), height);
    MyKernel1<<<100, 512>>>(devPtr, pitch, width, height);
}
void test2() {
    // Host code
    int width = 64, height = 64, depth = 64;
    cudaExtent extent = make_cudaExtent(width * sizeof(float),
                                        height, depth);
    cudaPitchedPtr devPitchedPtr;
    cudaMalloc3D(&devPitchedPtr, extent);
    MyKernel2<<<100, 512>>>(devPitchedPtr, width, height, depth);
}
int main(int argc, char* argv[]) {
    test1();
    test2();
    return 0;
}