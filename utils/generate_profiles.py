import json 
import numpy as np
import pprint



def generate_random_profile(template, ID = None):
    listing_file = json.load(open("config\dataset_listing.json"))
    template["ID"] = ID
    # Consumption

    template["consumption"]["type"] = np.random.choice(listing_file["consumption"])
    template["consumption"]["max_consumption"] = np.random.randint(10,21)
    template["consumption"]["cost_of_non_served_energy"] = np.random.random()/100+0.2
    # Generation
    template["generation"]["type"] = np.random.choice(listing_file["generation"])
    template["generation"]["max_generation"] = 0

    # Battery
    template["battery"]["power"] = np.random.randint(1,5)
    template["battery"]["enabled"] = bool(np.random.randint(0,2))
    template["battery"]["fcr_enabled"] = bool(np.random.randint(0,2))

    # Grid 
    template["grid"]["node"] = int(np.random.choice(listing_file["consumption_node"]))
    return template

if __name__ == "__main__":
    for i in range(20):
        json_file = json.load(open("config\household_profile\default_household.json"))  
        profile = generate_random_profile(json_file, ID = i)
        profile = json.dump(profile, open(f"config\household_profile/household_random_{i}.json", "w"), sort_keys=True, indent=4)

