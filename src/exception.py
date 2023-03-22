import sys

def error_message_detail(error, error_detail:sys):
    _, _, exec_tb = error_detail.exc_info()
    file_name = exec_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in Python script name [{file_name}] at Line: [{line_no}] \nerror message: [{message}]"