import sys # it gets variable info before hand by interpreter which we can use to show error.

def error_info_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() # we not need first two info
    file_name=exc_tb.tb_frame.f_code.co_filename # in exception docs
    error_message= "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)

    )
    return error_message

class CustomException:
    def __init___(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_info_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message