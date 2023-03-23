import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    _, _, exec_tb = error_detail.exc_info() # Execution tab
    file_name = exec_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in Python script name [{file_name}] at Line: [{exec_tb.tb_lineno}] \nerror message: [{str(error)}]"

    return error_message

class CustomException(Exception):
    def __init__(self, error, error_detail:sys):
        self.error_message = error_message_detail(error = error, error_detail = error_detail)
        super().__init__(error)

    def __str__(self):
        return self.error_message
    
if __name__ == '__main__':
    try: a = 1/0 
    except Exception as e:
        logging.info('Divide by 0 error')
        raise CustomException(e, sys)