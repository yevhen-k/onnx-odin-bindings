package yolov8_inference

import "core:c"

// when ODIN_OS == .Linux do foreign import cv "opencvbinding.a"
// when ODIN_OS == .Linux do foreign import cv "libopencvbinding.so"
when ODIN_OS == .Linux do foreign import cv "thirdparty/opencv/opencvbinding.a"

Mat :: distinct rawptr
VideoCapture :: distinct rawptr


ImageReadModes :: enum c.int {
	IMREAD_UNCHANGED           = -1, //!< If set, return the loaded image as is (with alpha channel,
	//!< otherwise it gets cropped). Ignore EXIF orientation.
	IMREAD_GRAYSCALE           = 0, //!< If set, always convert image to the single channel
	//!< grayscale image (codec internal conversion).
	IMREAD_COLOR               = 1, //!< If set, always convert image to the 3 channel BGR color image.
	IMREAD_ANYDEPTH            = 2, //!< If set, return 16-bit/32-bit image when the input has the
	//!< corresponding depth, otherwise convert it to 8-bit.
	IMREAD_ANYCOLOR            = 4, //!< If set, the image is read in any possible color format.
	IMREAD_LOAD_GDAL           = 8, //!< If set, use the gdal driver for loading the image.
	IMREAD_REDUCED_GRAYSCALE_2 = 16, //!< If set, always convert image to the single channel grayscale
	//!< image and the image size reduced 1/2.
	IMREAD_REDUCED_COLOR_2     = 17, //!< If set, always convert image to the 3 channel BGR color image
	//!< and the image size reduced 1/2.
	IMREAD_REDUCED_GRAYSCALE_4 = 32, //!< If set, always convert image to the single channel grayscale
	//!< image and the image size reduced 1/4.
	IMREAD_REDUCED_COLOR_4     = 33, //!< If set, always convert image to the 3 channel BGR color image
	//!< and the image size reduced 1/4.
	IMREAD_REDUCED_GRAYSCALE_8 = 64, //!< If set, always convert image to the single channel grayscale
	//!< image and the image size reduced 1/8.
	IMREAD_REDUCED_COLOR_8     = 65, //!< If set, always convert image to the 3 channel BGR color image
	//!< and the image size reduced 1/8.
	IMREAD_IGNORE_ORIENTATION  = 128, //!< If set, do not rotate the image
	//!< according to EXIF's orientation flag.
}

VideoCaptureAPIs :: enum c.int {
	CAP_ANY           = 0, //!< Auto detect == 0
	CAP_VFW           = 200, //!< Video For Windows (obsolete, removed)
	CAP_V4L           = 200, //!< V4L/V4L2 capturing support
	CAP_V4L2          = CAP_V4L, //!< Same as CAP_V4L
	CAP_FIREWIRE      = 300, //!< IEEE 1394 drivers
	CAP_FIREWARE      = CAP_FIREWIRE, //!< Same value as CAP_FIREWIRE
	CAP_IEEE1394      = CAP_FIREWIRE, //!< Same value as CAP_FIREWIRE
	CAP_DC1394        = CAP_FIREWIRE, //!< Same value as CAP_FIREWIRE
	CAP_CMU1394       = CAP_FIREWIRE, //!< Same value as CAP_FIREWIRE
	CAP_QT            = 500, //!< QuickTime (obsolete, removed)
	CAP_UNICAP        = 600, //!< Unicap drivers (obsolete, removed)
	CAP_DSHOW         = 700, //!< DirectShow (via videoInput)
	CAP_PVAPI         = 800, //!< PvAPI, Prosilica GigE SDK
	CAP_OPENNI        = 900, //!< OpenNI (for Kinect)
	CAP_OPENNI_ASUS   = 910, //!< OpenNI (for Asus Xtion)
	CAP_ANDROID       = 1000, //!< MediaNDK (API Level 21+) and NDK Camera (API level 24+) for Android
	CAP_XIAPI         = 1100, //!< XIMEA Camera API
	CAP_AVFOUNDATION  = 1200, //!< AVFoundation framework for iOS (OS X Lion will have the same API)
	CAP_GIGANETIX     = 1300, //!< Smartek Giganetix GigEVisionSDK
	CAP_MSMF          = 1400, //!< Microsoft Media Foundation (via videoInput). See platform specific notes above.
	CAP_WINRT         = 1410, //!< Microsoft Windows Runtime using Media Foundation
	CAP_INTELPERC     = 1500, //!< RealSense (former Intel Perceptual Computing SDK)
	CAP_REALSENSE     = 1500, //!< Synonym for CAP_INTELPERC
	CAP_OPENNI2       = 1600, //!< OpenNI2 (for Kinect)
	CAP_OPENNI2_ASUS  = 1610, //!< OpenNI2 (for Asus Xtion and Occipital Structure sensors)
	CAP_OPENNI2_ASTRA = 1620, //!< OpenNI2 (for Orbbec Astra)
	CAP_GPHOTO2       = 1700, //!< gPhoto2 connection
	CAP_GSTREAMER     = 1800, //!< GStreamer
	CAP_FFMPEG        = 1900, //!< Open and record video file or stream using the FFMPEG library
	CAP_IMAGES        = 2000, //!< OpenCV Image Sequence (e.g. img_%02d.jpg)
	CAP_ARAVIS        = 2100, //!< Aravis SDK
	CAP_OPENCV_MJPEG  = 2200, //!< Built-in OpenCV MotionJPEG codec
	CAP_INTEL_MFX     = 2300, //!< Intel MediaSDK
	CAP_XINE          = 2400, //!< XINE engine (Linux)
	CAP_UEYE          = 2500, //!< uEye Camera API
	CAP_OBSENSOR      = 2600, //!< For Orbbec 3D-Sensor device/module (Astra+, Femto, Astra2, Gemini2, Gemini2L, Gemini2XL, Femto Mega) attention: Astra2, Gemini2, and Gemini2L cameras currently only support Windows and Linux kernel versions no higher than 4.15, and higher versions of Linux kernel may have exceptions.
}

//! Color conversion codes
ColorConversionCodes :: enum c.int {
	COLOR_BGR2BGRA          = 0, //!< add alpha channel to RGB or BGR image
	COLOR_RGB2RGBA          = COLOR_BGR2BGRA,
	COLOR_BGRA2BGR          = 1, //!< remove alpha channel from RGB or BGR image
	COLOR_RGBA2RGB          = COLOR_BGRA2BGR,
	COLOR_BGR2RGBA          = 2, //!< convert between RGB and BGR color spaces (with or
	//!< without alpha channel)
	COLOR_RGB2BGRA          = COLOR_BGR2RGBA,
	COLOR_RGBA2BGR          = 3,
	COLOR_BGRA2RGB          = COLOR_RGBA2BGR,
	COLOR_BGR2RGB           = 4,
	COLOR_RGB2BGR           = COLOR_BGR2RGB,
	COLOR_BGRA2RGBA         = 5,
	COLOR_RGBA2BGRA         = COLOR_BGRA2RGBA,
	COLOR_BGR2GRAY          = 6, //!< convert between RGB/BGR and grayscale, @ref
	//!< color_convert_rgb_gray "color conversions"
	COLOR_RGB2GRAY          = 7,
	COLOR_GRAY2BGR          = 8,
	COLOR_GRAY2RGB          = COLOR_GRAY2BGR,
	COLOR_GRAY2BGRA         = 9,
	COLOR_GRAY2RGBA         = COLOR_GRAY2BGRA,
	COLOR_BGRA2GRAY         = 10,
	COLOR_RGBA2GRAY         = 11,
	COLOR_BGR2BGR565        = 12, //!< convert between RGB/BGR and BGR565 (16-bit images)
	COLOR_RGB2BGR565        = 13,
	COLOR_BGR5652BGR        = 14,
	COLOR_BGR5652RGB        = 15,
	COLOR_BGRA2BGR565       = 16,
	COLOR_RGBA2BGR565       = 17,
	COLOR_BGR5652BGRA       = 18,
	COLOR_BGR5652RGBA       = 19,
	COLOR_GRAY2BGR565       = 20, //!< convert between grayscale to BGR565 (16-bit images)
	COLOR_BGR5652GRAY       = 21,
	COLOR_BGR2BGR555        = 22, //!< convert between RGB/BGR and BGR555 (16-bit images)
	COLOR_RGB2BGR555        = 23,
	COLOR_BGR5552BGR        = 24,
	COLOR_BGR5552RGB        = 25,
	COLOR_BGRA2BGR555       = 26,
	COLOR_RGBA2BGR555       = 27,
	COLOR_BGR5552BGRA       = 28,
	COLOR_BGR5552RGBA       = 29,
	COLOR_GRAY2BGR555       = 30, //!< convert between grayscale and BGR555 (16-bit images)
	COLOR_BGR5552GRAY       = 31,
	COLOR_BGR2XYZ           = 32, //!< convert RGB/BGR to CIE XYZ, @ref
	//!< color_convert_rgb_xyz "color conversions"
	COLOR_RGB2XYZ           = 33,
	COLOR_XYZ2BGR           = 34,
	COLOR_XYZ2RGB           = 35,
	COLOR_BGR2YCrCb         = 36, //!< convert RGB/BGR to luma-chroma (aka YCC), @ref
	//!< color_convert_rgb_ycrcb "color conversions"
	COLOR_RGB2YCrCb         = 37,
	COLOR_YCrCb2BGR         = 38,
	COLOR_YCrCb2RGB         = 39,
	COLOR_BGR2HSV           = 40, //!< convert RGB/BGR to HSV (hue saturation value) with H range 0..180
	//!< if 8 bit image, @ref color_convert_rgb_hsv "color conversions"
	COLOR_RGB2HSV           = 41,
	COLOR_BGR2Lab           = 44, //!< convert RGB/BGR to CIE Lab, @ref
	//!< color_convert_rgb_lab "color conversions"
	COLOR_RGB2Lab           = 45,
	COLOR_BGR2Luv           = 50, //!< convert RGB/BGR to CIE Luv, @ref
	//!< color_convert_rgb_luv "color conversions"
	COLOR_RGB2Luv           = 51,
	COLOR_BGR2HLS           = 52, //!< convert RGB/BGR to HLS (hue lightness saturation)
	//!< with H range 0..180 if 8 bit image, @ref
	//!< color_convert_rgb_hls "color conversions"
	COLOR_RGB2HLS           = 53,
	COLOR_HSV2BGR           = 54, //!< backward conversions HSV to RGB/BGR with H range
	//!< 0..180 if 8 bit image
	COLOR_HSV2RGB           = 55,
	COLOR_Lab2BGR           = 56,
	COLOR_Lab2RGB           = 57,
	COLOR_Luv2BGR           = 58,
	COLOR_Luv2RGB           = 59,
	COLOR_HLS2BGR           = 60, //!< backward conversions HLS to RGB/BGR with H range
	//!< 0..180 if 8 bit image
	COLOR_HLS2RGB           = 61,
	COLOR_BGR2HSV_FULL      = 66, //!< convert RGB/BGR to HSV (hue saturation value) with H range 0..255
	//!< if 8 bit image, @ref color_convert_rgb_hsv "color conversions"
	COLOR_RGB2HSV_FULL      = 67,
	COLOR_BGR2HLS_FULL      = 68, //!< convert RGB/BGR to HLS (hue lightness
	//!< saturation) with H range 0..255 if 8 bit image,
	//!< @ref color_convert_rgb_hls "color conversions"
	COLOR_RGB2HLS_FULL      = 69,
	COLOR_HSV2BGR_FULL      = 70, //!< backward conversions HSV to RGB/BGR with H range
	//!< 0..255 if 8 bit image
	COLOR_HSV2RGB_FULL      = 71,
	COLOR_HLS2BGR_FULL      = 72, //!< backward conversions HLS to RGB/BGR with H range
	//!< 0..255 if 8 bit image
	COLOR_HLS2RGB_FULL      = 73,
	COLOR_LBGR2Lab          = 74,
	COLOR_LRGB2Lab          = 75,
	COLOR_LBGR2Luv          = 76,
	COLOR_LRGB2Luv          = 77,
	COLOR_Lab2LBGR          = 78,
	COLOR_Lab2LRGB          = 79,
	COLOR_Luv2LBGR          = 80,
	COLOR_Luv2LRGB          = 81,
	COLOR_BGR2YUV           = 82, //!< convert between RGB/BGR and YUV
	COLOR_RGB2YUV           = 83,
	COLOR_YUV2BGR           = 84,
	COLOR_YUV2RGB           = 85,

	//! YUV 4:2:0 family to RGB
	COLOR_YUV2RGB_NV12      = 90,
	COLOR_YUV2BGR_NV12      = 91,
	COLOR_YUV2RGB_NV21      = 92,
	COLOR_YUV2BGR_NV21      = 93,
	COLOR_YUV420sp2RGB      = COLOR_YUV2RGB_NV21,
	COLOR_YUV420sp2BGR      = COLOR_YUV2BGR_NV21,
	COLOR_YUV2RGBA_NV12     = 94,
	COLOR_YUV2BGRA_NV12     = 95,
	COLOR_YUV2RGBA_NV21     = 96,
	COLOR_YUV2BGRA_NV21     = 97,
	COLOR_YUV420sp2RGBA     = COLOR_YUV2RGBA_NV21,
	COLOR_YUV420sp2BGRA     = COLOR_YUV2BGRA_NV21,
	COLOR_YUV2RGB_YV12      = 98,
	COLOR_YUV2BGR_YV12      = 99,
	COLOR_YUV2RGB_IYUV      = 100,
	COLOR_YUV2BGR_IYUV      = 101,
	COLOR_YUV2RGB_I420      = COLOR_YUV2RGB_IYUV,
	COLOR_YUV2BGR_I420      = COLOR_YUV2BGR_IYUV,
	COLOR_YUV420p2RGB       = COLOR_YUV2RGB_YV12,
	COLOR_YUV420p2BGR       = COLOR_YUV2BGR_YV12,
	COLOR_YUV2RGBA_YV12     = 102,
	COLOR_YUV2BGRA_YV12     = 103,
	COLOR_YUV2RGBA_IYUV     = 104,
	COLOR_YUV2BGRA_IYUV     = 105,
	COLOR_YUV2RGBA_I420     = COLOR_YUV2RGBA_IYUV,
	COLOR_YUV2BGRA_I420     = COLOR_YUV2BGRA_IYUV,
	COLOR_YUV420p2RGBA      = COLOR_YUV2RGBA_YV12,
	COLOR_YUV420p2BGRA      = COLOR_YUV2BGRA_YV12,
	COLOR_YUV2GRAY_420      = 106,
	COLOR_YUV2GRAY_NV21     = COLOR_YUV2GRAY_420,
	COLOR_YUV2GRAY_NV12     = COLOR_YUV2GRAY_420,
	COLOR_YUV2GRAY_YV12     = COLOR_YUV2GRAY_420,
	COLOR_YUV2GRAY_IYUV     = COLOR_YUV2GRAY_420,
	COLOR_YUV2GRAY_I420     = COLOR_YUV2GRAY_420,
	COLOR_YUV420sp2GRAY     = COLOR_YUV2GRAY_420,
	COLOR_YUV420p2GRAY      = COLOR_YUV2GRAY_420,

	//! YUV 4:2:2 family to RGB
	COLOR_YUV2RGB_UYVY      = 107,
	COLOR_YUV2BGR_UYVY      = 108,
	// COLOR_YUV2RGB_VYUY = 109,
	// COLOR_YUV2BGR_VYUY = 110,
	COLOR_YUV2RGB_Y422      = COLOR_YUV2RGB_UYVY,
	COLOR_YUV2BGR_Y422      = COLOR_YUV2BGR_UYVY,
	COLOR_YUV2RGB_UYNV      = COLOR_YUV2RGB_UYVY,
	COLOR_YUV2BGR_UYNV      = COLOR_YUV2BGR_UYVY,
	COLOR_YUV2RGBA_UYVY     = 111,
	COLOR_YUV2BGRA_UYVY     = 112,
	// COLOR_YUV2RGBA_VYUY = 113,
	// COLOR_YUV2BGRA_VYUY = 114,
	COLOR_YUV2RGBA_Y422     = COLOR_YUV2RGBA_UYVY,
	COLOR_YUV2BGRA_Y422     = COLOR_YUV2BGRA_UYVY,
	COLOR_YUV2RGBA_UYNV     = COLOR_YUV2RGBA_UYVY,
	COLOR_YUV2BGRA_UYNV     = COLOR_YUV2BGRA_UYVY,
	COLOR_YUV2RGB_YUY2      = 115,
	COLOR_YUV2BGR_YUY2      = 116,
	COLOR_YUV2RGB_YVYU      = 117,
	COLOR_YUV2BGR_YVYU      = 118,
	COLOR_YUV2RGB_YUYV      = COLOR_YUV2RGB_YUY2,
	COLOR_YUV2BGR_YUYV      = COLOR_YUV2BGR_YUY2,
	COLOR_YUV2RGB_YUNV      = COLOR_YUV2RGB_YUY2,
	COLOR_YUV2BGR_YUNV      = COLOR_YUV2BGR_YUY2,
	COLOR_YUV2RGBA_YUY2     = 119,
	COLOR_YUV2BGRA_YUY2     = 120,
	COLOR_YUV2RGBA_YVYU     = 121,
	COLOR_YUV2BGRA_YVYU     = 122,
	COLOR_YUV2RGBA_YUYV     = COLOR_YUV2RGBA_YUY2,
	COLOR_YUV2BGRA_YUYV     = COLOR_YUV2BGRA_YUY2,
	COLOR_YUV2RGBA_YUNV     = COLOR_YUV2RGBA_YUY2,
	COLOR_YUV2BGRA_YUNV     = COLOR_YUV2BGRA_YUY2,
	COLOR_YUV2GRAY_UYVY     = 123,
	COLOR_YUV2GRAY_YUY2     = 124,
	// CV_YUV2GRAY_VYUY    = CV_YUV2GRAY_UYVY,
	COLOR_YUV2GRAY_Y422     = COLOR_YUV2GRAY_UYVY,
	COLOR_YUV2GRAY_UYNV     = COLOR_YUV2GRAY_UYVY,
	COLOR_YUV2GRAY_YVYU     = COLOR_YUV2GRAY_YUY2,
	COLOR_YUV2GRAY_YUYV     = COLOR_YUV2GRAY_YUY2,
	COLOR_YUV2GRAY_YUNV     = COLOR_YUV2GRAY_YUY2,

	//! alpha premultiplication
	COLOR_RGBA2mRGBA        = 125,
	COLOR_mRGBA2RGBA        = 126,

	//! RGB to YUV 4:2:0 family
	COLOR_RGB2YUV_I420      = 127,
	COLOR_BGR2YUV_I420      = 128,
	COLOR_RGB2YUV_IYUV      = COLOR_RGB2YUV_I420,
	COLOR_BGR2YUV_IYUV      = COLOR_BGR2YUV_I420,
	COLOR_RGBA2YUV_I420     = 129,
	COLOR_BGRA2YUV_I420     = 130,
	COLOR_RGBA2YUV_IYUV     = COLOR_RGBA2YUV_I420,
	COLOR_BGRA2YUV_IYUV     = COLOR_BGRA2YUV_I420,
	COLOR_RGB2YUV_YV12      = 131,
	COLOR_BGR2YUV_YV12      = 132,
	COLOR_RGBA2YUV_YV12     = 133,
	COLOR_BGRA2YUV_YV12     = 134,

	//! Demosaicing, see @ref color_convert_bayer "color conversions" for
	//! additional information
	COLOR_BayerBG2BGR       = 46, //!< equivalent to RGGB Bayer pattern
	COLOR_BayerGB2BGR       = 47, //!< equivalent to GRBG Bayer pattern
	COLOR_BayerRG2BGR       = 48, //!< equivalent to BGGR Bayer pattern
	COLOR_BayerGR2BGR       = 49, //!< equivalent to GBRG Bayer pattern
	COLOR_BayerRGGB2BGR     = COLOR_BayerBG2BGR,
	COLOR_BayerGRBG2BGR     = COLOR_BayerGB2BGR,
	COLOR_BayerBGGR2BGR     = COLOR_BayerRG2BGR,
	COLOR_BayerGBRG2BGR     = COLOR_BayerGR2BGR,
	COLOR_BayerRGGB2RGB     = COLOR_BayerBGGR2BGR,
	COLOR_BayerGRBG2RGB     = COLOR_BayerGBRG2BGR,
	COLOR_BayerBGGR2RGB     = COLOR_BayerRGGB2BGR,
	COLOR_BayerGBRG2RGB     = COLOR_BayerGRBG2BGR,
	COLOR_BayerBG2RGB       = COLOR_BayerRG2BGR, //!< equivalent to RGGB Bayer pattern
	COLOR_BayerGB2RGB       = COLOR_BayerGR2BGR, //!< equivalent to GRBG Bayer pattern
	COLOR_BayerRG2RGB       = COLOR_BayerBG2BGR, //!< equivalent to BGGR Bayer pattern
	COLOR_BayerGR2RGB       = COLOR_BayerGB2BGR, //!< equivalent to GBRG Bayer pattern
	COLOR_BayerBG2GRAY      = 86, //!< equivalent to RGGB Bayer pattern
	COLOR_BayerGB2GRAY      = 87, //!< equivalent to GRBG Bayer pattern
	COLOR_BayerRG2GRAY      = 88, //!< equivalent to BGGR Bayer pattern
	COLOR_BayerGR2GRAY      = 89, //!< equivalent to GBRG Bayer pattern
	COLOR_BayerRGGB2GRAY    = COLOR_BayerBG2GRAY,
	COLOR_BayerGRBG2GRAY    = COLOR_BayerGB2GRAY,
	COLOR_BayerBGGR2GRAY    = COLOR_BayerRG2GRAY,
	COLOR_BayerGBRG2GRAY    = COLOR_BayerGR2GRAY,

	//! Demosaicing using Variable Number of Gradients
	COLOR_BayerBG2BGR_VNG   = 62, //!< equivalent to RGGB Bayer pattern
	COLOR_BayerGB2BGR_VNG   = 63, //!< equivalent to GRBG Bayer pattern
	COLOR_BayerRG2BGR_VNG   = 64, //!< equivalent to BGGR Bayer pattern
	COLOR_BayerGR2BGR_VNG   = 65, //!< equivalent to GBRG Bayer pattern
	COLOR_BayerRGGB2BGR_VNG = COLOR_BayerBG2BGR_VNG,
	COLOR_BayerGRBG2BGR_VNG = COLOR_BayerGB2BGR_VNG,
	COLOR_BayerBGGR2BGR_VNG = COLOR_BayerRG2BGR_VNG,
	COLOR_BayerGBRG2BGR_VNG = COLOR_BayerGR2BGR_VNG,
	COLOR_BayerRGGB2RGB_VNG = COLOR_BayerBGGR2BGR_VNG,
	COLOR_BayerGRBG2RGB_VNG = COLOR_BayerGBRG2BGR_VNG,
	COLOR_BayerBGGR2RGB_VNG = COLOR_BayerRGGB2BGR_VNG,
	COLOR_BayerGBRG2RGB_VNG = COLOR_BayerGRBG2BGR_VNG,
	COLOR_BayerBG2RGB_VNG   = COLOR_BayerRG2BGR_VNG, //!< equivalent to RGGB Bayer pattern
	COLOR_BayerGB2RGB_VNG   = COLOR_BayerGR2BGR_VNG, //!< equivalent to GRBG Bayer pattern
	COLOR_BayerRG2RGB_VNG   = COLOR_BayerBG2BGR_VNG, //!< equivalent to BGGR Bayer pattern
	COLOR_BayerGR2RGB_VNG   = COLOR_BayerGB2BGR_VNG, //!< equivalent to GBRG Bayer pattern

	//! Edge-Aware Demosaicing
	COLOR_BayerBG2BGR_EA    = 135, //!< equivalent to RGGB Bayer pattern
	COLOR_BayerGB2BGR_EA    = 136, //!< equivalent to GRBG Bayer pattern
	COLOR_BayerRG2BGR_EA    = 137, //!< equivalent to BGGR Bayer pattern
	COLOR_BayerGR2BGR_EA    = 138, //!< equivalent to GBRG Bayer pattern
	COLOR_BayerRGGB2BGR_EA  = COLOR_BayerBG2BGR_EA,
	COLOR_BayerGRBG2BGR_EA  = COLOR_BayerGB2BGR_EA,
	COLOR_BayerBGGR2BGR_EA  = COLOR_BayerRG2BGR_EA,
	COLOR_BayerGBRG2BGR_EA  = COLOR_BayerGR2BGR_EA,
	COLOR_BayerRGGB2RGB_EA  = COLOR_BayerBGGR2BGR_EA,
	COLOR_BayerGRBG2RGB_EA  = COLOR_BayerGBRG2BGR_EA,
	COLOR_BayerBGGR2RGB_EA  = COLOR_BayerRGGB2BGR_EA,
	COLOR_BayerGBRG2RGB_EA  = COLOR_BayerGRBG2BGR_EA,
	COLOR_BayerBG2RGB_EA    = COLOR_BayerRG2BGR_EA, //!< equivalent to RGGB Bayer pattern
	COLOR_BayerGB2RGB_EA    = COLOR_BayerGR2BGR_EA, //!< equivalent to GRBG Bayer pattern
	COLOR_BayerRG2RGB_EA    = COLOR_BayerBG2BGR_EA, //!< equivalent to BGGR Bayer pattern
	COLOR_BayerGR2RGB_EA    = COLOR_BayerGB2BGR_EA, //!< equivalent to GBRG Bayer pattern

	//! Demosaicing with alpha channel
	COLOR_BayerBG2BGRA      = 139, //!< equivalent to RGGB Bayer pattern
	COLOR_BayerGB2BGRA      = 140, //!< equivalent to GRBG Bayer pattern
	COLOR_BayerRG2BGRA      = 141, //!< equivalent to BGGR Bayer pattern
	COLOR_BayerGR2BGRA      = 142, //!< equivalent to GBRG Bayer pattern
	COLOR_BayerRGGB2BGRA    = COLOR_BayerBG2BGRA,
	COLOR_BayerGRBG2BGRA    = COLOR_BayerGB2BGRA,
	COLOR_BayerBGGR2BGRA    = COLOR_BayerRG2BGRA,
	COLOR_BayerGBRG2BGRA    = COLOR_BayerGR2BGRA,
	COLOR_BayerRGGB2RGBA    = COLOR_BayerBGGR2BGRA,
	COLOR_BayerGRBG2RGBA    = COLOR_BayerGBRG2BGRA,
	COLOR_BayerBGGR2RGBA    = COLOR_BayerRGGB2BGRA,
	COLOR_BayerGBRG2RGBA    = COLOR_BayerGRBG2BGRA,
	COLOR_BayerBG2RGBA      = COLOR_BayerRG2BGRA, //!< equivalent to RGGB Bayer pattern
	COLOR_BayerGB2RGBA      = COLOR_BayerGR2BGRA, //!< equivalent to GRBG Bayer pattern
	COLOR_BayerRG2RGBA      = COLOR_BayerBG2BGRA, //!< equivalent to BGGR Bayer pattern
	COLOR_BayerGR2RGBA      = COLOR_BayerGB2BGRA, //!< equivalent to GBRG Bayer pattern

	//! RGB to YUV 4:2:2 family
	COLOR_RGB2YUV_UYVY      = 143,
	COLOR_BGR2YUV_UYVY      = 144,
	COLOR_RGB2YUV_Y422      = COLOR_RGB2YUV_UYVY,
	COLOR_BGR2YUV_Y422      = COLOR_BGR2YUV_UYVY,
	COLOR_RGB2YUV_UYNV      = COLOR_RGB2YUV_UYVY,
	COLOR_BGR2YUV_UYNV      = COLOR_BGR2YUV_UYVY,
	COLOR_RGBA2YUV_UYVY     = 145,
	COLOR_BGRA2YUV_UYVY     = 146,
	COLOR_RGBA2YUV_Y422     = COLOR_RGBA2YUV_UYVY,
	COLOR_BGRA2YUV_Y422     = COLOR_BGRA2YUV_UYVY,
	COLOR_RGBA2YUV_UYNV     = COLOR_RGBA2YUV_UYVY,
	COLOR_BGRA2YUV_UYNV     = COLOR_BGRA2YUV_UYVY,
	COLOR_RGB2YUV_YUY2      = 147,
	COLOR_BGR2YUV_YUY2      = 148,
	COLOR_RGB2YUV_YVYU      = 149,
	COLOR_BGR2YUV_YVYU      = 150,
	COLOR_RGB2YUV_YUYV      = COLOR_RGB2YUV_YUY2,
	COLOR_BGR2YUV_YUYV      = COLOR_BGR2YUV_YUY2,
	COLOR_RGB2YUV_YUNV      = COLOR_RGB2YUV_YUY2,
	COLOR_BGR2YUV_YUNV      = COLOR_BGR2YUV_YUY2,
	COLOR_RGBA2YUV_YUY2     = 151,
	COLOR_BGRA2YUV_YUY2     = 152,
	COLOR_RGBA2YUV_YVYU     = 153,
	COLOR_BGRA2YUV_YVYU     = 154,
	COLOR_RGBA2YUV_YUYV     = COLOR_RGBA2YUV_YUY2,
	COLOR_BGRA2YUV_YUYV     = COLOR_BGRA2YUV_YUY2,
	COLOR_RGBA2YUV_YUNV     = COLOR_RGBA2YUV_YUY2,
	COLOR_BGRA2YUV_YUNV     = COLOR_BGRA2YUV_YUY2,
	COLOR_COLORCVT_MAX      = 155,
}

//! interpolation algorithm
InterpolationFlags :: enum c.int {
	/** nearest neighbor interpolation */
	INTER_NEAREST       = 0,
	/** bilinear interpolation */
	INTER_LINEAR        = 1,
	/** bicubic interpolation */
	INTER_CUBIC         = 2,
	/** resampling using pixel area relation. It may be a preferred method for
  image decimation, as it gives moire'-free results. But when the image is
  zoomed, it is similar to the INTER_NEAREST method. */
	INTER_AREA          = 3,
	/** Lanczos interpolation over 8x8 neighborhood */
	INTER_LANCZOS4      = 4,
	/** Bit exact bilinear interpolation */
	INTER_LINEAR_EXACT  = 5,
	/** Bit exact nearest neighbor interpolation. This will produce same results
  as the nearest neighbor method in PIL, scikit-image or Matlab. */
	INTER_NEAREST_EXACT = 6,
	/** mask for interpolation codes */
	INTER_MAX           = 7,
	/** flag, fills all of the destination image pixels. If some of them
  correspond to outliers in the source image, they are set to zero */
	WARP_FILL_OUTLIERS  = 8,
	/** flag, inverse transformation

  For example, #linearPolar or #logPolar transforms:
  - flag is __not__ set: \f$dst( \rho , \phi ) = src(x,y)\f$
  - flag is set: \f$dst(x,y) = src( \rho , \phi )\f$
  */
	WARP_INVERSE_MAP    = 16,
}

DataTypes :: enum c.int {
	CV_8U  = 0,
	CV_8S  = 1,
	CV_16U = 2,
	CV_16S = 3,
	CV_32S = 4,
	CV_32F = 5,
	CV_64F = 6,
	CV_16F = 7,
}

MatView :: struct {
	rows:     u32,
	cols:     u32,
	channels: u32,
	type:     u32,
	dims:     u32,
	data:     [^]u8,
}

Detection :: struct {
	x:          c.float,
	y:          c.float,
	w:          c.float,
	h:          c.float,
	class_id:   c.float,
	confidence: c.float,
}

Detections :: struct {
	count:     c.int,
	detection: [^]Detection,
}

BatchDetections :: struct {
	batch_size: c.int,
	detections: [^]Detections,
}

Size :: struct {
	dim1: c.double,
	dim2: c.double,
}

Scalar :: struct {
	dim1: c.double,
	dim2: c.double,
	dim3: c.double,
}

Shape3i :: struct {
	dim1: c.int,
	dim2: c.int,
	dim3: c.int,
}

@(default_calling_convention = "c")
foreign cv {
	cv_new_mat :: proc() -> Mat ---
	cv_image_read :: proc(file: cstring, flags: ImageReadModes) -> Mat ---
	cv_cvt_color :: proc(src: Mat, dst: Mat, code: ColorConversionCodes, dstCn: c.int = 0) ---
	cv_mat_isempty :: proc(mat: Mat) -> c.bool ---
	cv_resize :: proc(src: Mat, dst: Mat, new_w: c.int, new_h: c.int, interpolation: InterpolationFlags) ---
	cv_named_window :: proc(name: cstring) ---
	cv_image_show :: proc(name: cstring, img: Mat) ---
	cv_wait_key :: proc(delay: c.int) -> c.int ---
	cv_destroy_window :: proc(name: cstring) ---
	cv_image_write :: proc(filename: cstring, img: Mat) -> c.bool ---
	cv_free_mem :: proc(data: rawptr) ---

	cv_new_videocapture :: proc() -> VideoCapture ---
	cv_videocapture_open :: proc(capture: VideoCapture, device_id, api_id: c.int) -> c.bool ---
	cv_videocapture_release :: proc(capture: VideoCapture) ---
	cv_videocapture_isopened :: proc(capture: VideoCapture) -> c.bool ---
	cv_videocapture_read :: proc(capture: VideoCapture, frame: Mat) -> c.bool ---

	cv_get_mat_view :: proc(mat: Mat) -> MatView ---

	cv_normalize :: proc(src: Mat, dst: Mat, rtype: c.int, alpha: c.double, beta: c.double) ---

	cv_blob_from_image :: proc(src: Mat, scalefactor: c.double, size: Size, mean: Scalar, swapRGB: c.bool, crop: c.bool, ddepth: DataTypes) -> Mat ---

	cv_parse_yolo_output :: proc(yolo_output: [^]c.float, yolo_output_shape: Shape3i, score_threshold: c.float, nms_threshold: c.float) -> BatchDetections ---
	cv_render_detection :: proc(image: Mat, detection: Detection, class_name: cstring) ---
}
