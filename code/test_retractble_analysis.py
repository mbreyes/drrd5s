#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:18:23 2024

@author: mbreyes
"""

import drrdTools as dr

def eliminate_events_between(event_list, between= [23,13], elim= [1,3]):
    '''
    Parameters
    ----------
    event_list : List of lists 
        List of [Time,event] for all events (matriz M from med2tec).
    between : List.
        List of two events that determine an inverval for which the events will
        be eliminated. For example, for the retractable lever codes (13 is 
        lever insertion and 23 lever retraction), we may want to eliminate all
        responses that happen when the lever are retracted. Hence we want to 
        eliminate responses between 23 and 13. In this case, all events in the 
        list elim (see below) found between a code 23 and a 13 will be
        eliminated from data. The default is [23,13].
    elim : List, optional
        List of codes that should be eliminated. The default is [1,3].

    Returns
    -------
    The event_list filtered.

    '''
    
    # Initialize variables
    filtered_list = []
    inside_target_range = False
    code1, code2 = between
    
    # Process the list
    for i, (timestamp, event_code) in enumerate(event_list):
        if event_code == code1:
            # Start of a target range
            inside_target_range = True
            filtered_list.append([timestamp, event_code])
        elif event_code == code2 and inside_target_range:
            # End of a target range
            inside_target_range = False
            filtered_list.append([timestamp, event_code])
        elif inside_target_range:
            # Within a target range, skip events of type 1 and 3
            if event_code in elim:
                continue
            else:
                filtered_list.append([timestamp, event_code])
        else:
            # Outside a target range, include all events
            filtered_list.append([timestamp, event_code])
    
    return filtered_list


# rats_g1 = [63,64,65,66,68,71,72,74] # grupo 5s
rats_g1 = [63,64,65,66,68,71,72,74] # grupo 5s
rats_g2 = [52,55,56,58,59,60,61,69] # grupo 10s

rats = rats_g1 + rats_g2
rats.sort()

for rat in rats:

    # M = dr.med2tec(f'../data/raw/AY4/AY40{rat}.001')
    
    # M = eliminate_events_between(M)
    
    print("\n\n\nrat", rat)
    # D = dr.individual_drrd(prefix='AY4', session=1, dataPath='../data/raw/AY4/')
    D = D = dr.drrd(prefix='AY4',animalID=rat,sessions=[1,2],\
                    dataPath='../data/raw/AY4/', events_to_eliminate=[5,9],
                    elimin_begin= False)

