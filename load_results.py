from tinydb import TinyDB, Query

# load database
db = TinyDB('db.json')
db_tests = db.table('tests')
db_tests_meta = db.table('tests_meta')
user = Query()

VIDEO_NAME = 'V_20210217_162057_N0.mp4'

meta = db_tests_meta.search(user.video_file_name == VIDEO_NAME)
print(meta)