def get_output_path_from_s3_url(s3_url):
  return s3_url.strip().replace("s3.us-east-2.amazonaws.com/beam-outputs/index.html#", "beam-outputs.s3.amazonaws.com/")
  
  
def print_things(data):
  print('the thing is:', data)
