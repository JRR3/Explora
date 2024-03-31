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

    def replace_treatment_labels(self):

        def fun(txt):
            if txt == "control":
                return "C"
            elif txt == "treated":
                return "T"
            else:
                raise ValueError("Unexpected state.")

        self.A.obs["state"] = self.A.obs["state"].apply(fun)

    def filter_cells_and_genes(self):
        sc.pp.filter_cells(self.A, min_counts=200)
        sc.pp.filter_genes(self.A, min_cells=1)

    def remove_high_mitochondrial_counts(self):
        self.A = self.A[self.A.obs.pct_counts_mt < 1, :]
    
    def compute_stats(self, adata=None):
        if adata is None:
            adata = self.A
        sc.pp.calculate_qc_metrics(adata,
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

    def remove_high_total_counts_by_state(self):
        mask_C = self.A.obs.state == "C"
        mask_T = ~mask_C
        x = self.A[mask_C].obs["total_counts"]
        q25, q75 = np.percentile(x, [25,75])
        iqr = q75 - q25
        U = 1.5 * iqr
        mask_C &= self.A.obs.total_counts < U

        x = self.A[mask_T].obs["total_counts"]
        q25, q75 = np.percentile(x, [25,75])
        iqr = q75 - q25
        U = 1.5 * iqr
        mask_T &= self.A.obs.total_counts < U

        mask = mask_C | mask_T

        self.A = self.A[mask]

    def plot_stats(self, label = ""):

        #Total counts (per cell)
        fig = px.box(self.A.obs,
                #y="state",
                x="total_counts",
                color="state",
                log_x=True,
        )
        fig.update_layout(xaxis_title="Total counts")
        txt = "i_box_total_counts_per_cell" + label + ".html"
        fig.write_html(txt)

        #Number of genes (per cell)
        fig = px.box(self.A.obs,
                #y="state",
                x="n_genes_by_counts",
                color="state",
                log_x=True,
        )
        fig.update_layout(xaxis_title="# of genes")
        txt = "i_box_n_genes" + label + ".html"
        fig.write_html(txt)

        #Percent of mitochondrial counts (per cell)
        fig = px.box(self.A.obs,
                #y="state",
                x="pct_counts_mt",
                color="state",
                log_x=False,
        )
        txt = "Mitochondrial counts (%)"
        fig.update_layout(xaxis_title=txt)

        txt = "i_box_pct_mito" + label + ".html"
        fig.write_html(txt)

        #MT% vs Total counts
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

        txt = "i_pct_mito_vs_total_counts" + label + ".html"
        fig.write_html(txt)


    def partition_into_states(self):
        self.C = self.A[self.A.obs.state == "C"].copy()
        self.T = self.A[self.A.obs.state == "T"].copy()
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
        c_labels = ["C"] * len(c_disp[0])
        t_labels = ["T"] * len(t_disp[0])
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
        mask = x < np.inf
        x = x[mask]
        y = y[mask]
        y -= x
        X = x.reshape(-1,1) ** 2
        reg_model = LinearRegression(fit_intercept=False)
        reg_model.fit(X, y)
        poly_C = [0, 1, reg_model.coef_[0]]
        print("C:", poly_C)

        #===========Prediction
        a = x.min()
        b = x.max()
        x_C = np.linspace(a,b,100)
        y_C = polyval(x_C, poly_C)
        mask = 0 < y_C
        x_C = x_C[mask]
        y_C = y_C[mask]

        #======================Treatment model
        x = t_disp[0]
        y = t_disp[1]
        mask = x < np.inf
        x = x[mask]
        y = y[mask]
        y -= x
        X = x.reshape(-1,1) ** 2
        reg_model = LinearRegression(fit_intercept=False)
        reg_model.fit(X, y)
        poly_T = [0, 1, reg_model.coef_[0]]
        print("T:", poly_T)

        #===========Prediction
        a = x.min()
        b = x.max()
        x_T = np.linspace(a,b,100)
        y_T = polyval(x_T, poly_T)
        mask = 0 < y_T
        x_T = x_T[mask]
        y_T = y_T[mask]

        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=c_disp[0],
                y=c_disp[1],
                mode="markers",
                name="C",
                ))

        fig.add_trace(
            go.Scatter(
                x=t_disp[0],
                y=t_disp[1],
                mode="markers",
                name="T",
                ))

        fig.add_trace(
            go.Scatter(
                x=x_C,
                y=y_C,
                mode="lines",
                name="a(C)=0.75",
                line=dict(color="blue"),
                ))

        fig.add_trace(
            go.Scatter(
                x=x_T,
                y=y_T,
                mode="lines",
                name="a(T)=2.38",
                line=dict(color="red"),
                ))


        fig.update_layout(
            # title=r"Var ~ mean + d*mean^2",
            xaxis_title="Mean(counts) for genes",
            yaxis_title="Var(counts) for genes",
        )

        use_log = True

        if use_log:
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
            fig.write_html("i_var_vs_mean_for_genes_log.html")
        else:
            fig.write_html("i_var_vs_mean_for_genes_lin.html")


        fig = px.scatter(df,
                x="mean",
                y="var",
                color="state",
                log_x=use_log,
                log_y=use_log,
                render_mode="svg",
        )
        x_txt = "Mean(counts) for genes"
        y_txt = "Var(counts) for genes"
        fig.update_layout(xaxis_title=x_txt,
                          yaxis_title=y_txt)
        if use_log:
            fig.write_html(
                "i_var_vs_mean_for_genes_px_log.html")
        else:
            fig.write_html(
                "i_var_vs_mean_for_genes_px_lin.html")

    def plot_counts_by_gene(self, label=""):
        #Total counts (per cell)
        
        self.compute_stats(self.C)
        self.compute_stats(self.T)


        df_C = self.C.var.copy()
        df_T = self.T.var.copy()

        df_C["state"] = "C"
        df_T["state"] = "T"

        df_C = df_C[["total_counts","state"]].values
        df_T = df_T[["total_counts","state"]].values

        df = np.concatenate((df_C,df_T),axis=0)
        df = pd.DataFrame(df,
                          columns=["total_counts","state"])


        df["total_counts"] = df["total_counts"].astype(float)
        df["total_counts"] += 1
        df["total_counts"] = np.log10(df["total_counts"])

        fig = px.histogram(df,
                #y="state",
                x="total_counts",
                color="state",
                #log_x=True,
                #log_y=True,
                nbins=50,
        )
        txt = "log10(Total counts per gene)"
        fig.update_layout(xaxis_title=txt)
        txt = "i_hist_total_counts_per_gene" + label + ".html"
        fig.write_html(txt)
                    


obj = Explora()
obj.replace_treatment_labels()
obj.find_mitochondrial_genes()
obj.compute_stats()
obj.plot_stats(label="_unfiltered")
obj.filter_cells_and_genes()
obj.remove_high_mitochondrial_counts()
obj.remove_high_total_counts_by_state()
obj.filter_cells_and_genes()
obj.compute_stats()
obj.plot_stats(label="_filtered")
obj.partition_into_states()
obj.plot_counts_by_gene()
#obj.plot_dispersion()
