// realesrgan implemented with ncnn library

#ifndef REALESRGAN_H
#define REALESRGAN_H

#ifdef WIN32
#ifdef LIBREALESRGAN_EXPORTS
#define LIBREALESRGAN_API __declspec(dllexport)
#else
#define LIBREALESRGAN_API __declspec(dllimport)
#endif
#else
#define LIBREALESRGAN_API
#endif

#include <filesystem>
#include <string>

// ncnn
#include "gpu.h"
#include "layer.h"
#include "net.h"

class LIBREALESRGAN_API RealESRGAN {
   public:
    RealESRGAN(int gpuid, bool tta_mode = false);
    ~RealESRGAN();

    int load(const std::filesystem::path &parampath, const std::filesystem::path &modelpath);

    int process(const ncnn::Mat &inimage, ncnn::Mat &outimage) const;

   public:
    // realesrgan parameters
    int scale;
    int tilesize;
    int prepadding;

   private:
    ncnn::Net net;
    ncnn::Pipeline *realesrgan_preproc;
    ncnn::Pipeline *realesrgan_postproc;
    ncnn::Layer *bicubic_2x;
    ncnn::Layer *bicubic_3x;
    ncnn::Layer *bicubic_4x;
    bool tta_mode;
};

#endif  // REALESRGAN_H
