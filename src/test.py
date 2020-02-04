# cat_cols = ["channelGrouping", "device.browser",
#                 "device.deviceCategory", "device.operatingSystem",
#                 "geoNetwork.city", "geoNetwork.continent",
#                 "geoNetwork.country", "geoNetwork.metro",
#                 "geoNetwork.networkDomain", "geoNetwork.region",
#                 "geoNetwork.subContinent",
#                 "trafficSource.medium",
#                 "trafficSource.source"]
#
#
# to_drop = ["socialEngagementType", 'device.browserVersion', 'device.browserSize', 'device.flashVersion',
#                'device.language',
#                'device.mobileDeviceBranding', 'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName',
#                'device.mobileDeviceModel',
#                'device.mobileInputSelector', 'device.operatingSystemVersion', 'device.screenColors',
#                'device.screenResolution',
#                'geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.networkLocation',
#                'trafficSource.adwordsClickInfo.criteriaParameters', 'trafficSource.adwordsClickInfo.gclId',
#                'trafficSource.campaign',
#                'trafficSource.adwordsClickInfo.page', 'trafficSource.referralPath',
#                'trafficSource.adwordsClickInfo.slot',
#                'trafficSource.adContent', 'trafficSource.keyword', 'trafficSource.adwordsClickInfo.adNetworkType',
#                'trafficSource.campaignCode', 'totals.bounces', 'totals.newVisits', 'totals.visits',
#                'trafficSource.isTrueDirect',
#                'trafficSource.adwordsClickInfo.isVideoAd', 'totals.visits']
#
# num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',
#             'totals.newVisits']
#
#
# def common(a,b):
#     c = [value for value in a if value in b]
#     return c
#
#
# d = common(cat_cols, to_drop)
# print(d)
#
#
# f = common(num_cols, to_drop)
# print(f)
#

import pandas as pd
import numpy as np
df = pd.DataFrame([('bird', 389.0),
                   ('bird', 24.0),
                   ('mammal', 80.5),
                   ('mammal', np.nan)],
                  # index=['falcon', 'parrot', 'lion', 'monkey'],
                  columns=('class', 'max_speed'))
print(df)
df.set_index('class', append=False, drop=True, inplace=True)
print(df)
