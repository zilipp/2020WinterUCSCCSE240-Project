import plotly.graph_objs as go
import plotly.offline as py
from plotly import subplots


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
        # Continent
        cnt_srs = self.train_df.groupby('trafficSource.source')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
        cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'green')
        trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'green')
        trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'green')

        # Sub-continent
        cnt_srs = self.train_df.groupby('trafficSource.medium')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
        cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace4 = horizontal_bar_chart(cnt_srs["count"], 'purple')
        trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"], 'purple')
        trace6 = horizontal_bar_chart(cnt_srs["mean"], 'purple')

        # Creating two subplots
        fig = subplots.make_subplots(rows=2, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15,
                                  subplot_titles=["Traffic Source - Count", "Traffic Source - Non-zero Revenue Count",
                                                  "Traffic Source - Mean Revenue",
                                                  "Traffic Source Medium - Count",
                                                  "Traffic Source Medium - Non-zero Revenue Count",
                                                  "Traffic Source Medium - Mean Revenue"
                                                  ])

        fig.append_trace(trace1, 1, 1)
        fig.append_trace(trace2, 1, 2)
        fig.append_trace(trace3, 1, 3)
        fig.append_trace(trace4, 2, 1)
        fig.append_trace(trace5, 2, 2)
        fig.append_trace(trace6, 2, 3)

        fig['layout'].update(height=1000, width=1200, paper_bgcolor='rgb(233,233,233)', title="Traffic Source Plots")
        py.plot(fig, filename='../graphs/traffic-source-plots.html', auto_open=False)

    def plot_diff_device_importance(self):
        # Device Browser
        cnt_srs = self.train_df.groupby('device.browser')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
        cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
        trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')
        trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

        # Device Category
        cnt_srs = self.train_df.groupby('device.deviceCategory')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
        cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace4 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(71, 58, 131, 0.8)')
        trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(71, 58, 131, 0.8)')
        trace6 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(71, 58, 131, 0.8)')

        # Operating system
        cnt_srs = self.train_df.groupby('device.operatingSystem')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
        cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace7 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(246, 78, 139, 0.6)')
        trace8 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(246, 78, 139, 0.6)')
        trace9 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(246, 78, 139, 0.6)')

        # Creating two subplots
        fig = subplots.make_subplots(rows=3, cols=3, vertical_spacing=0.04,
                                     subplot_titles=["Device Browser - Count",
                                                     "Device Browser - Non-zero Revenue Count",
                                                     "Device Browser - Mean Revenue",
                                                     "Device Category - Count",
                                                     "Device Category - Non-zero Revenue Count",
                                                     "Device Category - Mean Revenue",
                                                     "Device OS - Count", "Device OS - Non-zero Revenue Count",
                                                     "Device OS - Mean Revenue"])

        fig.append_trace(trace1, 1, 1)
        fig.append_trace(trace2, 1, 2)
        fig.append_trace(trace3, 1, 3)
        fig.append_trace(trace4, 2, 1)
        fig.append_trace(trace5, 2, 2)
        fig.append_trace(trace6, 2, 3)
        fig.append_trace(trace7, 3, 1)
        fig.append_trace(trace8, 3, 2)
        fig.append_trace(trace9, 3, 3)

        fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233,233,233)', title="Device Plots")
        py.plot(fig, filename='../graphs/device-plots.html', auto_open=False)

    def plot_diff_geo_importance(self):
        # Continent
        cnt_srs = self.train_df.groupby('geoNetwork.continent')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
        cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(58, 71, 80, 0.6)')
        trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(58, 71, 80, 0.6)')
        trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(58, 71, 80, 0.6)')

        # Sub-continent
        cnt_srs = self.train_df.groupby('geoNetwork.subContinent')['totals.transactionRevenue'].agg(
            ['size', 'count', 'mean'])
        cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace4 = horizontal_bar_chart(cnt_srs["count"], 'orange')
        trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"], 'orange')
        trace6 = horizontal_bar_chart(cnt_srs["mean"], 'orange')

        # Network domain
        cnt_srs = self.train_df.groupby('geoNetwork.networkDomain')['totals.transactionRevenue'].agg(
            ['size', 'count', 'mean'])
        cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
        cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
        trace7 = horizontal_bar_chart(cnt_srs["count"].head(10), 'blue')
        trace8 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'blue')
        trace9 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'blue')

        # Creating two subplots
        fig = subplots.make_subplots(rows=3, cols=3, vertical_spacing=0.08, horizontal_spacing=0.15,
                                     subplot_titles=["Continent - Count", "Continent - Non-zero Revenue Count",
                                                     "Continent - Mean Revenue",
                                                     "Sub Continent - Count", "Sub Continent - Non-zero Revenue Count",
                                                     "Sub Continent - Mean Revenue",
                                                     "Network Domain - Count",
                                                     "Network Domain - Non-zero Revenue Count",
                                                     "Network Domain - Mean Revenue"])

        fig.append_trace(trace1, 1, 1)
        fig.append_trace(trace2, 1, 2)
        fig.append_trace(trace3, 1, 3)
        fig.append_trace(trace4, 2, 1)
        fig.append_trace(trace5, 2, 2)
        fig.append_trace(trace6, 2, 3)
        fig.append_trace(trace7, 3, 1)
        fig.append_trace(trace8, 3, 2)
        fig.append_trace(trace9, 3, 3)

        fig['layout'].update(height=1500, width=1200, paper_bgcolor='rgb(233,233,233)', title="Geography Plots")
        py.plot(fig, filename='../graphs/geo-plots.html', auto_open=False)

    def plot_revenue_count_with_time(self):
        # size includes NaN values, count does not:
        cnt_srs = self.train_df.groupby('date')['totals.transactionRevenue'].agg(['size', 'count'])
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
