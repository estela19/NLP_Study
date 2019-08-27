#레벤슈타인 거리 구하기
def calc_distance(a, b):
  """레벤슈타인 거리 계산하기"""
  if a == b : return 0
  a_len = len(a)
  b_len = len(b)
  if a == "" : return b_len
  if b == "" : return a_len
  #2차원 표 생성 및 초기화
  matrix = [[] for i in range(a_len + 1)]
  for i in range(a_len + 1):
    matrix[i] = [0 for j in range(b_len + 1)] 
  for i in range (a_len + 1):
    matrix[i][0] = i
  for j in range (b_len + 1):
    matrix[0][j] = j
  
  #표 채우기
  for i in range(1, a_len + 1):
    ac = a[i - 1]
    for j in range(1, b_len + 1):
      bc = b[j - 1]
      cost = 0 if (ac == bc) else 1
      matrix[i][j] = min([
          matrix[i - 1][j] + 1, # 문자삽입
          matrix[i][j - 1] + 1, # 문자제거
          matrix[i - 1][j - 1] + cost  #문자변경
      ])
  return matrix[a_len][b_len]
