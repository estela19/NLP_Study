from konlpy.tag import Twitter

twitter = Twitter()
malist = twitter.pos("고전 역학에 따르면, 물체의 크기에 관계없이 초기 운동상태를 정확히 알 수 있다면 일정한 시간 후의 물체의 상태는 정확히 측정될 수 있으며, 배타적인 두 개의 상태가 공존할 수 없다.", norm = True, stem = True)
print(malist)

"""
# result
[('고전', 'Noun'), ('역학', 'Noun'), ('에', 'Josa'), ('따르다', 'Verb'), (',', 'Punctuation'), ('물체', 'Noun'), ('의', 'Josa'), ('크기', 'Noun'), ('에', 'Josa'), ('관계없이', 'Adverb'), ('초기', 'Noun'), ('운동', 'Noun'), ('상태', 'Noun'), ('를', 'Josa'), ('정확하다', 'Adjective'), ('알', 'Noun'), ('수', 'Noun'), ('있다', 'Adjective'), ('일정하다', 'Adjective'), ('시간', 'Noun'), ('후의', 'Noun'), ('물체', 'Noun'), ('의', 'Josa'), ('상태', 'Noun'), ('는', 'Josa'), ('정확하다', 'Adjective'), ('측정', 'Noun'), ('되다', 'Verb'), ('수', 'Noun'), ('있다', 'Adjective'), (',', 'Punctuation'), ('배타', 'Noun'), ('적', 'Suffix'), ('인', 'Josa'), ('두', 'Noun'), ('개', 'Noun'), ('의', 'Josa'), ('상태', 'Noun'), ('가', 'Josa'), ('공존', 'Noun'), ('하다', 'Verb'), ('수', 'Noun'), ('없다', 'Adjective'), ('.', 'Punctuation')]
"""
