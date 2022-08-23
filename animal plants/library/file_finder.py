import os
files = []
def get_list_files(path):
	files = []
	for (dirpath,dirnames,filenames) in os.walk(path):
		if (len(filenames) > 0):
			for i in filenames:
				files.append(os.path.join(dirpath,i))
	return files
		

