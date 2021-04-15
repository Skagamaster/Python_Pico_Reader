import pico_reader
import numpy as np
import awkward as ak

pr = pico_reader.PicoDST()
pr.import_data("st_physics_21250020_raw_6500001.picoDst.root")

def basic_filter_1cm(events):
    return events.v_r < 1  # Only get events within 1 cm of center

def basic_filter_2cm(events):
    return events.v_r < 2  # Only get events within 2 cm of center

def basic_vertex_cut(events):
    select = events.v_r <= 2.0  # transverse vertex within 2cm
    select = select & np.abs(events.v_z) <= 30.0  # and a z cut at +- 30.0 cm
    return select

print(max(pr.v_r))

cut_events_1cm = pico_reader.Event_Cuts(pr, criteria=basic_filter_1cm)
cut_events_2cm = pico_reader.Event_Cuts(pr, criteria=basic_filter_2cm)
print("From", pr.num_events, ",", cut_events_1cm.num_events, "events have v_r < 1 cm")
print("From", pr.num_events, ",", cut_events_2cm.num_events, "events have v_r < 2 cm")

print("Events with v_r < 1 cm:")
print(cut_events_1cm.v_r)
print("The East/West Variable for acceptable events")
print(cut_events_1cm.epd_hits.EW)
print("The number of events left in the E/W display, should be equal to the number of events \
with a v_r < 1 cm")
print(len(cut_events_1cm.epd_hits.EW))
# Testing taking sum of cut events
print("testing numpy functions on cut events")
print(np.sum(cut_events_1cm.epd_hits.EW))
