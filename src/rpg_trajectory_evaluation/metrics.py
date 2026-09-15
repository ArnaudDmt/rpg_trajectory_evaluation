#!/usr/bin/env python2

# The translation error is exported both as a norm and split into the horizontal norm and the
# vertical component: the two are driven by different physics (odometry drift in the plane vs
# contact-height bookkeeping), so a tuning has to be scored on them separately.
kRelMetrics = ['rel_trans', 'rel_trans_perc', 'rel_trans_x_y_norm', 'rel_trans_z', 'rel_yaw',
               'rel_rot', 'rel_gravity', 'rel_rot_deg_per_m']
kRelMetricLables = ['trans', 'trans_perc', 'trans_x_y_norm', 'trans_z', 'yaw',
                    'rot', 'gravity', 'rot_deg_per_m']
