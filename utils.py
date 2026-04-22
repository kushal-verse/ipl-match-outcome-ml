TEAMS = sorted([
    'Chennai Super Kings',
    'Deccan Chargers',
    'Delhi Capitals',
    'Gujarat Lions',
    'Gujarat Titans',
    'Kochi Tuskers Kerala',
    'Kolkata Knight Riders',
    'Lucknow Super Giants',
    'Mumbai Indians',
    'Pune Warriors',
    'Punjab Kings',
    'Rajasthan Royals',
    'Rising Pune Supergiants',
    'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad',
])

HOME_GROUNDS = {
    'Mumbai Indians'             : 'Wankhede Stadium',
    'Chennai Super Kings'        : 'MA Chidambaram Stadium',
    'Royal Challengers Bengaluru': 'M Chinnaswamy Stadium',
    'Kolkata Knight Riders'      : 'Eden Gardens',
    'Delhi Capitals'             : 'Arun Jaitley Stadium',
    'Punjab Kings'               : 'Punjab Cricket Association IS Bindra Stadium',
    'Rajasthan Royals'           : 'Sawai Mansingh Stadium',
    'Sunrisers Hyderabad'        : 'Rajiv Gandhi International Stadium',
    'Gujarat Titans'             : 'Narendra Modi Stadium',
    'Lucknow Super Giants'       : 'BRSABV Ekana Cricket Stadium',
    'Deccan Chargers'            : 'Rajiv Gandhi International Stadium',
    'Kochi Tuskers Kerala'       : 'Jawaharlal Nehru Stadium',
    'Pune Warriors'              : 'Maharashtra Cricket Association Stadium',
    'Rising Pune Supergiants'    : 'Maharashtra Cricket Association Stadium',
    'Gujarat Lions'              : 'Saurashtra Cricket Association Stadium',
}

def prompt_team(prompt, exclude=None, valid_set=None):
    pool = valid_set if valid_set else TEAMS
    while True:
        name  = input(prompt).strip()
        match = next((t for t in pool if t.lower() == name.lower()), None)
        if not match:
            label = "Valid options" if not valid_set else "Must be one of the playing teams"
            print(f"  Unrecognised team. {label}:")
            for t in pool:
                print(f"        {t}")
            continue
        if exclude and match.lower() == exclude.lower():
            print(f"  Cannot be the same as the other team ({exclude}).")
            continue
        return match

def prompt_venue(prompt, valid_venues):
    while True:
        value = input(prompt).strip()
        if not value:
            print("  Input cannot be empty.")
            continue
        match = next((v for v in valid_venues if v.lower() == value.lower()), None)
        if match:
            return match
        print("  Unrecognised venue. Known venues:")
        for v in sorted(valid_venues):
            print(f"        {v}")

def prompt_str(prompt, valid=None):
    while True:
        value = input(prompt).strip()
        if not value:
            print("  Input cannot be empty.")
            continue
        if valid and value.lower() not in [v.lower() for v in valid]:
            print(f"  Must be one of: {valid}")
            continue
        return value

def prompt_int(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = int(input(prompt).strip())
            if min_val is not None and value < min_val:
                print(f"  Must be >= {min_val}.")
                continue
            if max_val is not None and value > max_val:
                print(f"  Must be <= {max_val}.")
                continue
            return value
        except ValueError:
            print("  Please enter a whole number.")
