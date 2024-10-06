// realesrgan implemented with ncnn library

#ifndef REALESRGAN_H
#define REALESRGAN_H

#ifdef WIN32
#ifdef LIBREALSERGAN_EXPORTS
#define LIBREALSERGAN_API __declspec(dllexport)
#else
#define LIBREALSERGAN_API __declspec(dllimport)
#endif
#else
#define LIBREALSERGAN_API
#endif

#include <string>

// ncnn
#include "gpu.h"
#include "layer.h"
#include "net.h"

class LIBREALSERGAN_API RealESRGAN {
   public:
    RealESRGAN(int gpuid, bool tta_mode = false);
    ~RealESRGAN();

#if _WIN32
    int load(const std::wstring &parampath, const std::wstring &modelpath);
#else
    int load(const std::string &parampath, const std::string &modelpath);
#endif

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
