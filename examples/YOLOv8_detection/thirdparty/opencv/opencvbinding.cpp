#include "opencvbinding.h"
#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

Mat cv_new_mat() { return new cv::Mat(); }

Mat cv_image_read(const char *file, int flags) {
  cv::Mat image = cv::imread(file, flags);
  return new cv::Mat(image);
}

void cv_cvt_color(Mat src, Mat dst, int code, int dstCn = 0) {
  cv::Mat *src_mat = static_cast<cv::Mat *>(src);
  cv::Mat *dst_mat = static_cast<cv::Mat *>(dst);
  cv::cvtColor(*src_mat, *dst_mat, code, dstCn);
}

bool cv_mat_isempty(Mat mat) {
  cv::Mat *m = static_cast<cv::Mat *>(mat);
  return m->empty();
}

void cv_resize(Mat src, Mat dst, int new_w, int new_h, int interpolation) {
  cv::Mat *src_mat = static_cast<cv::Mat *>(src);
  cv::Mat *dst_mat = static_cast<cv::Mat *>(dst);
  cv::Size dsize{new_w, new_h};
  cv::resize(*src_mat, *dst_mat, dsize, 0, 0, interpolation);
}

void cv_named_window(const char *name) {
  cv::namedWindow(name);
}

void cv_image_show(const char *name, Mat img) {
  cv::Mat *image = static_cast<cv::Mat *>(img);
  cv::imshow(name, *image);
}

int cv_wait_key(int delay) { return cv::waitKey(delay); }

void cv_destroy_window(const char *name) {
  cv::destroyWindow(name);
}

bool cv_image_write(const char *filename, Mat img) {
  cv::Mat *image = static_cast<cv::Mat *>(img);
  return cv::imwrite(filename, *image);
}

void cv_free_mem(void *data) { free(data); }

VideoCapture cv_new_videocapture() { return new cv::VideoCapture(); }

bool cv_videocapture_open(VideoCapture cap, int device_id, int api_id) {
  cv::VideoCapture *capture = static_cast<cv::VideoCapture *>(cap);
  return capture->open(device_id, api_id);
}

void cv_videocapture_release(VideoCapture cap) {
  cv::VideoCapture *capture = static_cast<cv::VideoCapture *>(cap);
  capture->release();
}

bool cv_videocapture_isopened(VideoCapture cap) {
  cv::VideoCapture *capture = static_cast<cv::VideoCapture *>(cap);
  return capture->isOpened();
}

bool cv_videocapture_read(VideoCapture cap, Mat frame) {
  cv::Mat *image = static_cast<cv::Mat *>(frame);
  cv::VideoCapture *capture = static_cast<cv::VideoCapture *>(cap);
  return capture->read(*image);
}

MatView cv_get_mat_view(Mat mat) {
  cv::Mat *m = static_cast<cv::Mat *>(mat);
  return MatView{m->rows, m->cols, m->channels(), m->type(), m->dims, m->data};
}

void cv_normalize(Mat src, Mat dst, int rtype, double alpha, double beta) {
  cv::Mat *src_mat = static_cast<cv::Mat *>(src);
  cv::Mat *dst_mat = static_cast<cv::Mat *>(dst);
  src_mat->convertTo(*dst_mat, rtype, alpha, beta);
}

Mat cv_blob_from_image(Mat src, double scalefactor, const Size size, const Scalar mean, bool swapRGB, bool crop, int ddepth) {
  cv::Mat *src_mat = static_cast<cv::Mat *>(src);
  
  cv::Mat blob = cv::dnn::blobFromImage(
      *src_mat,
      scalefactor,
      cv::Size(size.dim1, size.dim2),
      cv::Scalar(mean.dim1, mean.dim2, mean.dim3),
      swapRGB,
      crop,
      ddepth
    );
  // -1 -1 1 5 4 [3 x 1]
  // std::cout << blob.rows << " " << blob.cols << " " << blob.channels() << " " << blob.type() << " " << blob.dims << " " << blob.size() << "\n";
  return new cv::Mat(blob);
}

BatchDetections cv_parse_yolo_output(float* yolo_output, Shape3i yolo_output_shape, float score_threshold, float nms_threshold) {
  // https://docs.opencv.org/3.4/d3/d63/classcv_1_1Mat.html#a5fafc033e089143062fd31015b5d0f40
  int sizes[3] = {yolo_output_shape.dim1, yolo_output_shape.dim2, yolo_output_shape.dim3};
  cv::Mat yolo_res(
    3, // ndims
    sizes,
    CV_32F,
    yolo_output
  );
  
  // std::cout << yolo_res.rows << " " << yolo_res.cols << " " << yolo_res.channels() << " " << yolo_res.type() << " " << yolo_res.dims << " " << yolo_res.size() << "\n";
  // -1 -1 1 5 3 [84 x 1]
  
  // for(int i=0; i<yolo_res.dims;i++){
  //   std::cout << yolo_res.size[i] << " ";
  // }
  // std::cout << "\n";
  // 1 84 8400

  BatchDetections batch_detections;
  batch_detections.batch_size = yolo_res.size[0];
  batch_detections.detections = (Detections*)malloc(sizeof(Detections)*batch_detections.batch_size);

  // iterating over batches
  for(int batch=0; batch < batch_detections.batch_size; batch++) {
    cv::Mat tmp(yolo_res.size[1], yolo_res.size[2], CV_32F, yolo_res.ptr<float>(batch));
    cv::Mat inference_res = tmp.t();
    // std::cout << inference_res.rows << " " << inference_res.cols << " " << inference_res.channels() << " " << inference_res.type() << " " << inference_res.dims << " " << inference_res.size() << "\n";
    // 8400 84 1 5 2 [84 x 8400]
    
    std::vector<float> class_scores;
    std::vector<cv::Rect> boxes;
    std::vector<float> class_ids;

    // iterating over rows of prediction for an image/frame
    for(int rows = 0; rows < inference_res.rows; rows++) {

      float *classes_scores = inference_res.ptr<float>(rows, 4);
      cv::Mat scores(1, 80, CV_32FC1, classes_scores);
      cv::Point class_id;
      double class_score;
      cv::minMaxLoc(scores, 0, &class_score, 0, &class_id);

      // filter by threshold
      if (class_score > score_threshold) {
        float w = *inference_res.ptr<float>(rows, 2);
        float h = *inference_res.ptr<float>(rows, 3);
        float x = *inference_res.ptr<float>(rows, 0) - (0.5 * w);
        float y = *inference_res.ptr<float>(rows, 1) - (0.5 * h);
        class_scores.push_back(class_score);
        class_ids.push_back(class_id.x);
        // std::cout << "> DETECTION: " << x << " " << y << " " << w << " " << h << "\n";
        boxes.emplace_back(x, y, w, h);
      }
    }
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, nms_result);
    
    Detections detections;
    detections.count = nms_result.size();
    detections.detection = (Detection*)malloc(sizeof(Detection) * detections.count);

    for(int i = 0; i < detections.count; ++i) {
      int idx = nms_result[i];
      detections.detection[i].x = boxes[idx].x;
      detections.detection[i].y = boxes[idx].y;
      detections.detection[i].w = boxes[idx].width;
      detections.detection[i].h = boxes[idx].height;
      detections.detection[i].class_id = class_ids[idx];
      detections.detection[i].confidence = class_scores[idx];
      // std::cout << "> DETECTION: " << detections.detection[i].x << " " << detections.detection[i].y << " " << detections.detection[i].w << " " << detections.detection[i].h << " " << detections.detection[i].class_id << "\n";
    }

    batch_detections.detections[batch] = detections;
  }

  return batch_detections;

}

void cv_render_detection(Mat image, Detection detection, const char* class_name) {
  cv::Mat *mat = static_cast<cv::Mat *>(image);
  
  cv::rectangle(
    *mat, 
    cv::Point(detection.x, detection.y), 
    cv::Point(detection.x + detection.w, detection.y + detection.h), 
    cv::Scalar(0, 0, 255),
    2 // thickness
  );

  cv::putText(
    *mat,
    class_name,
    cv::Point(detection.x, detection.y),
    cv::FONT_HERSHEY_COMPLEX,
    1.0, // fontScale
    cv::Scalar(0, 0, 255),
    2, // thickness
    cv::LINE_AA // lineType
  );

}

