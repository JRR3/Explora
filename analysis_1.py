import scanpy as sc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyval
from numpy.polynomial.polynomial import Polynomial
import statsmodels.formula.api as smf
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.rcParams["figure.dpi"] = 600
# mpl.use("agg")

class Explora:
    def __init__(self):
        #The filtering took place before the h5ad file
        #was created.
        self.A = sc.read_h5ad("matrix.h5ad")

    def find_mitochondrial_genes(self):
        self.A.var["mt"] = self.A.var_names.str.startswith(
            "MT-")

    def filter_cells_and_genes(self):
        sc.pp.filter_cells(self.A, min_counts=200)
        sc.pp.filter_genes(self.A, min_cells=1)

    def remove_high_mitochondrial_counts(self):
        self.A = self.A[self.A.obs.pct_counts_mt < 1, :]
    
    def compute_stats(self):
        sc.pp.calculate_qc_metrics(self.A,
                                   qc_vars=["mt"],
                                   percent_top=None,
                                   log1p=False,
                                   inplace=True,
        )

    def remove_high_total_counts(self):
        x = self.A.obs["total_counts"]
        q25, q75 = np.percentile(x, [25,75])
        iqr = q75 - q25
        U = 1.5 * iqr
        self.A = self.A[self.A.obs.total_counts < U, :]

    def plot_stats(self):

        #Total counts
        fig = px.box(self.A.obs,
                #y="state",
                x="total_counts",
                color="state",
                log_x=True,
        )
        fig.update_layout(xaxis_title="Total counts")
        fig.write_html("i_box_total_counts.html")

        #Number of genes
        fig = px.box(self.A.obs,
                #y="state",
                x="n_genes_by_counts",
                color="state",
                log_x=True,
        )
        fig.update_layout(xaxis_title="# of genes")
        fig.write_html("i_box_n_genes.html")

        #Percent of mitochondrial counts
        fig = px.box(self.A.obs,
                #y="state",
                x="pct_counts_mt",
                color="state",
                log_x=False,
        )
        txt = "Mitochondrial counts (%)"
        fig.update_layout(xaxis_title=txt)
        fig.write_html("i_box_pct_mito.html")

        #
        fig = px.scatter(self.A.obs,
                x="total_counts",
                y="pct_counts_mt",
                color="state",
                log_x=True,
                render_mode="svg",
        )
        x_txt = "Total counts"
        y_txt = "Mitochondrial counts (%)"
        fig.update_layout(xaxis_title=x_txt,
                          yaxis_title=y_txt)
        fig.write_html("i_pct_mito_vs_total_counts.html")

    def partition_into_states(self):
        self.C = self.A[self.A.obs.state == "control"].copy()
        self.T = self.A[self.A.obs.state == "treated"].copy()
        print(self.C)
        print(self.T)

    def get_n_genes(self):
        return self.A.var.shape[0]

    def compute_dispersion(self, adata):
        gene_mean=adata.X.mean(axis=0)
        sq = adata.X.copy()
        sq.data **= 2
        gene_sq = sq.mean(axis=0)
        gene_mean = np.asarray(gene_mean).reshape(-1)
        gene_sq  = np.asarray(gene_sq).reshape(-1)
        gene_var = gene_sq - gene_mean**2
        mask = 0 < gene_mean
        return (gene_mean[mask], gene_var[mask])

    def plot_dispersion(self):
        c_disp = self.compute_dispersion(self.C)
        t_disp = self.compute_dispersion(self.T)
        c_labels = ["control"] * len(c_disp[0])
        t_labels = ["treated"] * len(t_disp[0])
        state = c_labels + t_labels
        g_mean = np.concatenate((c_disp[0], t_disp[0]))
        g_var = np.concatenate((c_disp[1], t_disp[1]))
        df=pd.DataFrame({"mean":g_mean,
                         "var":g_var,
                         "state":state,
                         })
        #======================Control model
        x = c_disp[0]
        y = c_disp[1]
        mask = x < 10
        mask = x < np.inf
        x = x[mask]
        y = y[mask]
        y_c = y - x
        # poly = PolynomialFeatures(degree=2,
        #                           include_bias=True)
        # poly_features = poly.fit_transform(x.reshape(-1,1))
        X = x.reshape(-1,1) ** 2
        reg_model = LinearRegression(fit_intercept=False)
        reg_model.fit(X, y_c)
        poly_c = [0, 1, reg_model.coef_[0]]
        # print(reg_model.coef_)
        # print(reg_model.intercept_)
        print(poly_c)

        a = x.min()
        b = x.max()
        print("values:", a, b)
        x_control = np.linspace(a,b,100)
        y_control = polyval(x_control, poly_c)
        mask = 0 < y_control
        x_control = x_control[mask]
        y_control = y_control[mask]

        fig = go.Figure()


        fig.add_trace(
            go.Scattergl(
                x=c_disp[0],
                y=c_disp[1],
                mode="markers",
                name="control",
                ))

        fig.add_trace(
            go.Scattergl(
                x=t_disp[0],
                y=t_disp[1],
                mode="markers",
                name="treated",
                ))

        fig.add_trace(
            go.Scattergl(
                x=x_control,
                y=y_control,
                mode="lines",
                name="Reg. (control)",
                ))

        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

        fig.write_html("i_var_vs_mean_for_genes.html")
        return


        fig = px.scatter(df,
                x="mean",
                y="var",
                color="state",
                log_x=True,
                log_y=True,
                render_mode="svg",
        )
        x_txt = "Mean(counts) for genes"
        y_txt = "Var(counts) for genes"
        fig.update_layout(xaxis_title=x_txt,
                          yaxis_title=y_txt)
        fig.write_html("i_var_vs_mean_for_genes.html")

                    


obj = Explora()
obj.filter_cells_and_genes()
obj.find_mitochondrial_genes()
obj.compute_stats()
obj.remove_high_mitochondrial_counts()
obj.remove_high_total_counts()
obj.filter_cells_and_genes()
obj.compute_stats()
obj.plot_stats()
obj.partition_into_states()
obj.plot_dispersion()
