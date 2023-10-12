/*
 * @Author: JHC521PJJ 
 * @Date: 2023-08-15 12:27:31 
 * @Last Modified by:   JHC521PJJ 
 * @Last Modified time: 2023-08-15 12:27:31 
 */

#include "onnxLog.h"


namespace ocr
{
OnnxLog log_info(LogLevel::LOG_LEVEL_INFO);
OnnxLog log_warning(LogLevel::LOG_LEVEL_WARNING);
OnnxLog log_error(LogLevel::LOG_LEVEL_ERROR);
}