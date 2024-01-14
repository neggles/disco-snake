import re

re_mention_resp = re.compile(r"(@\w+)\b", re.I)

test_str = "@Nyamo What's wrong with being a little quirky and fun? 😜"
user_mention = "<@123456>"

response = test_str
for user in re_mention_resp.finditer(response):
    user_re = re.compile(re.escape(user), re.I)
    response = user_re.sub(user_mention, response)
print(response)
