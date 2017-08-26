# Function to convert an age expressed in pcw, months, or years into post-conception weeks
# Potentially useful if your data is not all in the same units
# The function assumes that the units on the original data are either 'pcw', 'mos', or 'yrs'

def age_in_pcw(age):
    num, unit = age.split(' ')
    if unit == 'pcw':
        return int(num)
    elif unit == 'mos':
        return int(4.34524 * float(num)) + 40
    elif unit == 'yrs':
        return int(52.1429 * float(num)) + 40
    else:
        return 'What are these units???'

# Example

# ages_in = ['12 pcw', '13 pcw', '16 pcw', '17 pcw', '19 pcw', '21 pcw', '24 pcw',
#        '26 pcw', '37 pcw', '4 mos', '10 mos', '1 yrs', '2 yrs', '3 yrs',
#        '8 yrs', '11 yrs', '13 yrs', '18 yrs', '19 yrs', '21 yrs', '30 yrs',
#        '36 yrs', '37 yrs', '40 yrs', '5 days']
# ages_out = []
# for age in ages_in:
#     ages_out += [age_in_pcw(age)]
# print(ages_out)  
