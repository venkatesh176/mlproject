import sys
from src.logger import logging
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_line = exc_tb.tb_lineno
    error_msg = str(error)
    error_message = (f"error occured in python script name {file_name} line number {error_line} error message {error_msg}")
    return error_message

class CustomException(Exception):
    def __init__(self,error,error_detail:sys):
        super().__init__(str(error))
        self.error_message = error_message_detail(error,error_detail)

    def __str__(self):
        return self.error_message
    

             