import re

# variable for regex search phone numbers
string = "My number is 000-000-0000."
mo = re.search("\d\d\d-\d\d\d-\d\d\d\d", string)

print("Phone number found: " + mo[1])
