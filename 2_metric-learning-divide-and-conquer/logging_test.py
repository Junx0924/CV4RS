import logging

pj_base_path = "/home/users/p/paka0401/CV4RS/CV4RS/2_metric-learning-divide-and-conquer/"
DIYlogger = logging.getLogger()
DIYlogger.setLevel(logging.INFO)
_FMT_STRING = '[%(levelname)s:%(asctime)s] %(message)s'
_DATE_FMT = '%Y-%m-%d %H:%M:%S'
file_handler = logging.FileHandler(pj_base_path + "log/diy_log_test.txt", mode='w+')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(_FMT_STRING, datefmt=_DATE_FMT))
DIYlogger.addHandler(file_handler)
DIYlogger.info("If the logger works, this should appear in a file named diy_log_test.txt")
