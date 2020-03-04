import plotly.graph_objs as go
import plotly.offline as py
from plotly import subplots
import datetime
import pandas as pd
import numpy as np

# credit to:
# parse JSON:
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook
# baseline model : lgbm
# https://www.kaggle.com/zili100097/simple-exploration-baseline-ga-customer-revenue/edit
# missing value and unique value processing:
# https://www.kaggle.com/kabure/exploring-the-consumer-patterns-ml-pipeline


def horizontal_bar_chart(cnt_srs, color):
    trace = go.Bar(
        y=cnt_srs.index[::-1],
        x=cnt_srs.values[::-1],
        showlegend=False,
        orientation='h',
        marker=dict(
            color=color,
        ),
    )
    return trace


def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace


class Visualization:
    def __init__(self, train_df):
        self.train_df = train_df

    def plot_diff_traffic_importance(self):
        # trafficSourcesource
        cnt_srs = self.train_df.groupby('trafficSourcesource')['totalstransactionRevenue'].agg(
            ['size', self.count_nonzero])
        cnt_srs.columns = ["count", "count of non-zero revenue"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(135,206,250, 0.6)')

        # trafficSourcemedium
        cnt_srs = self.train_df.groupby('trafficSourcemedium')['totalstransactionRevenue'].agg(
            ['size', self.count_nonzero])
        cnt_srs.columns = ["count", "count of non-zero revenue"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"], 'rgba(135,206,250, 0.6)')

        fig = subplots.make_subplots(rows=1, cols=2, vertical_spacing=0.08, horizontal_spacing=0.15,
                                     subplot_titles=["Source", "Medium"])

        fig.append_trace(trace2, 1, 1)
        fig.append_trace(trace5, 1, 2)
        fig['layout'].update(margin=dict(l=140), height=400, width=1000, paper_bgcolor='rgb(233,233,233)',
                             title="Traffic Source")
        py.plot(fig, filename='../graphs/traffic-source-plots.html', auto_open=False)

    def plot_diff_device_importance(self):
        # Device Browser
        cnt_srs = self.train_df.groupby('devicebrowser')['totalstransactionRevenue'].agg(['size', self.count_nonzero])
        cnt_srs.columns = ["count", "count of non-zero revenue"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(135,206,250, 0.6)')

        # Device Category
        cnt_srs = self.train_df.groupby('devicedeviceCategory')['totalstransactionRevenue'].agg(
            ['size', self.count_nonzero])
        cnt_srs.columns = ["count", "count of non-zero revenue"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(135,206,250, 0.6)')

        # Operating system
        cnt_srs = self.train_df.groupby('deviceoperatingSystem')['totalstransactionRevenue'].agg(
            ['size', self.count_nonzero])
        cnt_srs.columns = ["count", "count of non-zero revenue"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace8 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(135,206,250, 0.6)')

        # Creating two subplots
        fig = subplots.make_subplots(rows=1, cols=3, vertical_spacing=0.04,
                                     subplot_titles=["Device Browser", "Device Category", "Device OS "])

        fig.append_trace(trace2, 1, 1)
        fig.append_trace(trace5, 1, 2)
        fig.append_trace(trace8, 1, 3)

        fig['layout'].update(margin=dict(l=140), height=400, width=1000, paper_bgcolor='rgb(233,233,233)',
                             title="Device")
        py.plot(fig, filename='../graphs/device-plots.html', auto_open=False)

    def plot_diff_geo_importance(self):
        # Continent
        cnt_srs = self.train_df.groupby('geoNetworkcontinent')['totalstransactionRevenue'].agg(
            ['size', 'self.count_nonzero'])
        cnt_srs.columns = ["count", "count of non-zero revenue"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(134, 226, 213, 1)')

        # Sub-continent
        cnt_srs = self.train_df.groupby('geoNetworksubContinent')['totalstransactionRevenue'].agg(
            ['size', self.count_nonzero])
        cnt_srs.columns = ["count", "count of non-zero revenue"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"], 'rgba(134, 226, 213, 1)')

        # Network domain
        cnt_srs = self.train_df.groupby('geoNetworknetworkDomain')['totalstransactionRevenue'].agg(
            ['size', self.count_nonzero])
        cnt_srs.columns = ["count", "count of non-zero revenue"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace8 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(134, 226, 213, 1)')

        # Creating two subplots
        fig = subplots.make_subplots(rows=1, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15,
                                     subplot_titles=["Continent ", "Sub Continent", "Network Domain "])

        fig.append_trace(trace2, 1, 1)
        fig.append_trace(trace5, 1, 2)
        fig.append_trace(trace8, 1, 3)

        fig['layout'].update(height=400, width=1000, paper_bgcolor='rgb(233,233,233)', title="Geography")
        py.plot(fig, filename='../graphs/geo-plots.html', auto_open=False)

    def plot_revenue_count_with_time(self):
        self.train_df['date'] = self.train_df['date'].apply(
            lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
        self.train_df["totalstransactionRevenue"] = self.train_df["totalstransactionRevenue"].astype \
            (float)

        # size includes NaN values, count does not:
        cnt_srs = self.train_df.groupby('date')['totalstransactionRevenue'].agg(['size', self.count_nonzero])
        cnt_srs.columns = ["count", "count of non-zero revenue"]
        cnt_srs = cnt_srs.sort_index()
        # cnt_srs.index = cnt_srs.index.astype('str')
        trace1 = scatter_plot(cnt_srs["count"], 'red')
        trace2 = scatter_plot(cnt_srs["count of non-zero revenue"], 'blue')

        fig = subplots.make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                                     subplot_titles=["Date - Count", "Date - Non-zero Revenue count"])
        fig.append_trace(trace1, 1, 1)
        fig.append_trace(trace2, 2, 1)
        fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
        py.plot(fig, filename='../graphs/date-plots.html', auto_open=False)

    def plot_visit_importance(self):
        # Page views
        cnt_srs = self.train_df.groupby('totalspageviews')['totalstransactionRevenue'].agg(
            ['size', self.count_nonzero])
        cnt_srs.columns = ["count", "count of non-zero revenue"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(60), 'rgba(134, 226, 213, 1)')

        # Hits
        cnt_srs = self.train_df.groupby('totalshits')['totalstransactionRevenue'].agg(
            ['size', self.count_nonzero])
        cnt_srs.columns = ["count", "count of non-zero revenue"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace4 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(60), 'rgba(134, 226, 213, 1)')

        # Creating two subplots
        fig = subplots.make_subplots(rows=1, cols=2, vertical_spacing=0.08, horizontal_spacing=0.15,
                                     subplot_titles=["Total Pageviews", "Total Hits"])

        fig.append_trace(trace2, 1, 1)
        fig.append_trace(trace4, 1, 2)

        fig['layout'].update(height=400, width=1000, paper_bgcolor='rgb(233,233,233)', title="Visitor Profile")
        py.plot(fig, filename='../graphs/visitor-profile-plots.html', auto_open=False)

    def count_nonzero(self, x):
        return np.count_nonzero(x)


if __name__ == '__main__':
    df_train = pd.read_csv("../data/train_full.csv")
    vis = Visualization(df_train)

    # a) with time
    vis.plot_revenue_count_with_time()
    # b) difference of device
    vis.plot_diff_device_importance()
    # c) traffic source
    vis.plot_diff_traffic_importance()
    # d) geo distribution
    vis.plot_diff_geo_importance()
    # e) visit profile
    vis.plot_visit_importance()
