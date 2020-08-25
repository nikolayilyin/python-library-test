def get_person_request(req_df, person_id):
    street_vehicle_id = "body-%s" % (person_id)
#     filtered_df = req_df[(req_df['streetVehicle_0_id'] == street_vehicle_id) | (req_df['streetVehicle_0_id'] == person_id) |
#                          (req_df['streetVehicle_1_id'] == street_vehicle_id) | (req_df['streetVehicle_1_id'] == person_id) |
#                          (req_df['streetVehicle_2_id'] == street_vehicle_id) | (req_df['streetVehicle_2_id'] == person_id)]
    filtered_df = req_df[(req_df['streetVehicle_0_id'].str.contains(person_id)) |
                         (req_df['streetVehicle_1_id'].str.contains(person_id)) |
                         (req_df['streetVehicle_2_id'].str.contains(person_id))]
    filtered_df = filtered_df.sort_values(by=['departureTime'])
    return filtered_df