#include "onnxLog.h"


namespace ocr
{
OnnxLog log_info(LogLevel::LOG_LEVEL_INFO);
OnnxLog log_warning(LogLevel::LOG_LEVEL_WARNING);
OnnxLog log_error(LogLevel::LOG_LEVEL_ERROR);
}