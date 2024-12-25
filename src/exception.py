import sys
from src.logger import logging

def error_message_func(error,error_details:sys):
    a,b,exc_tb=error_details.exc_info()
    error_message = "Error occured in scipt [{0}] at line number [{1}].\n Error message is [{2}].".format(
        exc_tb.tb_frame.f_code.co_filename,
        exc_tb.tb_lineno,
        str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.message = error_message_func(error_message,error_details)

    def __str__(self):
        return self.message
    

if __name__=="__main__":
    logging.info("Checking for execption handling")
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by zero occur")
        raise CustomException(e,sys)
