import os
import scanpy as sc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyval
from toomanycells import TooManyCells as tmc
# from io import BytesIO
import scipy.sparse as sp
from scipy.io import mmwrite

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["figure.dpi"] = 600
mpl.use("agg")

class Explora:
    #====================================================
    def __init__(self):
        #The filtering took place before the h5ad file
        #was created.
        # self.A = sc.read_h5ad("matrix.h5ad")
        self.source = ("/home/javier/Documents/persister"
             "/data/experiment_1/"
             "with_mouse_data/data/human")

        self.fig_out = ("/home/javier/Documents/persister"
             "/data/experiment_1/"
             "with_mouse_data/exploration/figures")

        os.makedirs(self.fig_out, exist_ok=True)

    #====================================================
    def load_mtx_object(self):
        self.A = sc.read_10x_mtx(self.source)
        print(self.A)

    #====================================================
    def load_h5ad_object(self):
        p = os.path.join(self.source, "matrix.h5ad")
        self.A = sc.read_h5ad(p)
        print(self.A)

    #====================================================
    def convert_mtx_object_to_h5ad(self):
        p = os.path.join(self.source, "matrix.h5ad")
        self.A.write_h5ad(p)

    #====================================================
    def load_metadata(self):
        p = os.path.join(self.source, "metadata.csv")
        self.df_meta = pd.read_csv(p, header=0, index_col=0)
        print(self.df_meta)
        mask = self.df_meta.index.isin(self.A.obs_names)
        print("Are all indices in the h5ad object:")
        if not mask.all():
            raise ValueError("Mismatch between metadata.")

        self.df_meta = self.df_meta.loc[self.A.obs_names]

        self.A.obs["state"] = self.df_meta["type"]
        self.A.obs["metaDiapause"] = self.df_meta["DTP.score"]


    #====================================================
    def find_mitochondrial_genes(self):
        self.A.var["mt"] = self.A.var_names.str.startswith(
            "MT-")

    #====================================================
    def replace_treatment_labels(self):

        def fun(txt):
            if txt == "control":
                return "C"
            elif txt == "treated":
                return "T"
            else:
                raise ValueError("Unexpected state.")

        self.A.obs["state"] = self.A.obs["state"].apply(fun)

    #====================================================
    def filter_cells_and_genes(self):
        print("Filtering cells and genes ...")
        # sc.pp.filter_cells(self.A, min_counts=200)

        sc.pp.filter_cells(self.A,
                           min_counts=50,
                           inplace=True)

        sc.pp.filter_genes(self.A,
                           min_cells=1,
                           inplace=True)

    #====================================================
    def remove_high_mitochondrial_counts(self):
        # We remove cells whose mitochondrial counts 
        # exceed the 1 percent of total counts.
        # Because this is a fixed threshold, there 
        # is no need to separate by state.

        mask = self.A.obs.pct_counts_mt >= 1
        mito_outliers = mask.sum()
        txt = (f"We have {mito_outliers} cells "
               "above 1% MT counts.")
        print(txt)

        self.A = self.A[self.A.obs.pct_counts_mt < 1, :]
    
    #====================================================
    def compute_stats(self, adata=None):
        print("Computing stats ...")
        if adata is None:
            sc.pp.calculate_qc_metrics(self.A,
                                    qc_vars=["mt"],
                                    percent_top=None,
                                    log1p=False,
                                    inplace=True,
            )
        else:
            sc.pp.calculate_qc_metrics(adata,
                                    qc_vars=["mt"],
                                    percent_top=None,
                                    log1p=False,
                                    inplace=True,
            )

    #====================================================
    def remove_high_total_counts(self):
        x = self.A.obs["total_counts"]
        q25, q75 = np.percentile(x, [25,75])
        iqr = q75 - q25
        U = 1.5 * iqr
        mask = self.A.obs.total_counts < q75 + U
        self.A = self.A[mask,:]

    #====================================================
    def remove_high_total_counts_by_state(self):
        mask_C = self.A.obs.state == "C"
        mask_T = ~mask_C
        x = self.A[mask_C].obs["total_counts"]
        q25, q75 = np.percentile(x, [25,75])
        print(f"Control:[{q25},{q75}]")
        iqr = q75 - q25
        U = 1.5 * iqr
        upper_bound = q75 + U
        print(f"Control IQR*1.5: {U}")
        print(f"Control upper bound: {upper_bound}")
        mask_C &= self.A.obs.total_counts < upper_bound

        x = self.A[mask_T].obs["total_counts"]
        q25, q75 = np.percentile(x, [25,75])
        print(f"Treatment:[{q25},{q75}]")
        iqr = q75 - q25
        U = 1.5 * iqr
        upper_bound = q75 + U
        print(f"Treatment IQR*1.5: {U}")
        print(f"Treatment upper bound: {upper_bound}")
        mask_T &= self.A.obs.total_counts < upper_bound

        mask = mask_C | mask_T

        self.A = self.A[mask]

    #====================================================
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


    #=====================================================
    def partition_into_states(self):
        self.C = self.A[self.A.obs.state == "C"].copy()
        self.T = self.A[self.A.obs.state == "T"].copy()
        print(self.C)
        print(self.T)

    #=====================================================
    def get_n_genes(self):
        return self.A.var.shape[0]

    #=====================================================
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

    #=====================================================
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

    #=====================================================
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


    #=====================================================
    def generate_features_and_barcodes(self):
        df = self.T.obs.copy()
        df = df.reset_index()
        df = df["barcode"]
        df.to_csv("barcodes.tsv", index=False)

    #====================================================
    def create_diapause_matrix(self):

        # self.compute_stats(self.T)
        # sc.pp.filter_cells(self.T,
        #                    min_counts=1,
        #                    inplace=True)

        # sc.pp.filter_genes(self.T,
        #                    min_cells=1,
        #                    inplace=True)

        #Prepare Z
        self.Z = self.A.copy()
        sc.pp.normalize_total(self.Z,
                              target_sum=10000,
                              inplace=False)
        sc.pp.scale(self.Z, copy=False)

        #Prepare A
        sc.pp.normalize_total(self.A,
                              target_sum=100,
                              inplace=True)
        sc.pp.log1p(self.A, copy=False)
        
        vec = np.zeros(self.Z.X.shape[0])
        up_reg = vec * 0
        down_reg = vec * 0
        up_count = 0
        down_count = 0
        df_signature = pd.read_csv(
            "diapause_signature.csv",header=0)
        # print(self.T.var)
        G = df_signature["Gene"]
        W = df_signature["Weight"]
        for gene, weight in zip(G, W):
            if gene not in self.Z.var.index:
                continue
            col_index = self.Z.var.index.get_loc(gene)
            if sp.issparse(self.Z.X):
                gene_col = self.Z.X.getcol(col_index)
                gene_col = np.squeeze(gene_col.toarray())
            else:
                gene_col = self.Z.X[:,col_index]

            if 0 < weight:
                up_reg += gene_col
                up_count += 1
            else:
                down_reg += gene_col
                down_count += 1
        
        total_counts = up_count + down_count

        chrDiap = 1 * up_reg - 1 * down_reg
        chrDiap /= total_counts

        up_factor = down_count / total_counts
        down_factor = up_count / total_counts

        modified_total_counts = 2 * up_count * down_count
        modified_total_counts /= total_counts
        
        check = up_factor*up_count + down_factor*down_count

        print(f"{up_count=}")
        print(f"{down_count=}")
        print(f"{total_counts=}")
        print(f"{modified_total_counts=}")
        print(f"{check=}")
        print(f"{up_factor=}")
        print(f"{down_factor=}")

        up_reg_mean   = up_reg / up_count
        down_reg_mean = down_reg / down_count

        wDiap = up_factor * up_reg - down_factor * down_reg
        wDiap /= modified_total_counts

        # print(wDiap)


        metDiap = self.A.obs["metaDiapause"]

        m = np.vstack((up_reg_mean,
                       down_reg_mean,
                       -down_reg_mean,
                       metDiap,
                       chrDiap,
                       wDiap))

        self.A.obs["wDiapause"] = wDiap
        self.Z.obs["wDiapause"] = wDiap

        self.A.obs["UpReg"] = up_reg_mean
        self.Z.obs["UpReg"] = up_reg_mean

        self.A.obs["DownReg"] = -down_reg_mean
        self.Z.obs["DownReg"] = -down_reg_mean

        print(self.A.obs["UpReg"].describe())
        print(self.A.obs["DownReg"].describe())

        m = m.astype(np.float32)

        self.diapause_mtx = sp.coo_matrix(m)

        # target = BytesIO()
        # target = "./treatment_data/matrix.mtx"
        # mmwrite(target, coo_matrix(m))

    #====================================================
    def create_visualization(self):
        tmc_obj = tmc(self.A, "treatment_tmc_outputs")

        tmc_obj.run_spectral_clustering()
        tmc_obj.store_outputs()

        tmc_obj.create_data_for_tmci(
            list_of_genes = ["UpReg",
                             "DownReg",
                             "negDownReg",
                             "metaDiapause",
                             "chrDiapause",
                             "wDiapause",
                             ],
            create_matrix=False)
        
        mtx_path = os.path.join(
            tmc_obj.tmci_mtx_dir, "matrix.mtx")
        mmwrite(mtx_path, self.diapause_mtx)

        #MAC
        # tmci_dir = ("/Users/javier/Documents/"
        #             "too-many-cells-interactive")

        mtx_dir = ("/home/javier/Documents/persister"
                   "/data/experiment_1/with_mouse_data"
                   "/exploration/treatment_tmc_outputs"
                   "/tmci_mtx_data")

        # mtx_dir = ""

        tmci_dir = ("/home/javier/Documents/"
                   "repos/too-many-cells-interactive")

        tmc_obj.visualize_with_tmc_interactive(
            tmci_dir,
            "state",
            2234,
            include_matrix_data = True,
            tmci_mtx_dir=mtx_dir,
            )

    #====================================================
    def compute_pca(self, obj, label):
        fig, ax = plt.subplots()
        sc.tl.pca(obj)

        colors=["state", "UpReg", "DownReg", "wDiapause"]
        ftype = ".png"

        for c in colors:
            fig, ax = plt.subplots()
            sc.pl.pca(obj,
                      color=[c],
                      ax = ax,
                      show=False)

            fname = label
            fname += "_"
            fname += c
            fname += ftype
            fname = os.path.join(self.fig_out, fname)
            fig.savefig(fname, bbox_inches="tight")

    #====================================================
    def pca_step(self):
        self.compute_pca(self.A, "pca_ct_norm")
        self.compute_pca(self.Z, "pca_z_norm")

    #====================================================
    def compute_diff_map(self, obj, label):
        sc.pp.neighbors(obj,
                        n_neighbors=10,
                        use_rep="X",
                        method="gauss",
                        metric="euclidean",
                        )
        sc.tl.diffmap(obj, n_comps=5)
        obj.obsm["X_diffmap_"] = obj.obsm["X_diffmap"][:, 1:]

        colors=["state", "UpReg", "DownReg", "wDiapause"]
        ftype = ".png"
        for c in colors:
            fig, ax = plt.subplots()
            sc.pl.embedding(obj,
                            "diffmap_",
                            color=[c],
                            ax = ax,
                            show=False)

            fname = label
            fname += "_"
            fname += c
            fname += ftype
            fname = os.path.join(self.fig_out, fname)
            fig.savefig(fname, bbox_inches="tight")


    #====================================================
    def diff_map_step(self):
        self.compute_diff_map(self.A, "diff_ct_norm")
        self.compute_diff_map(self.Z, "diff_z_norm")

    #====================================================
    def full_tmc(self):
        #Prepare Z
        self.Z = self.A.copy()
        sc.pp.normalize_total(self.Z,
                              target_sum=100,
                              inplace=False)
        sc.pp.scale(self.Z, copy=False)

        #Prepare A
        sc.pp.normalize_total(self.A,
                              target_sum=100,
                              inplace=True)
        sc.pp.log1p(self.A, copy=False)

        tmc_obj = tmc(self.A, "treatment_tmc_outputs")
        tmc_obj.run_spectral_clustering()
        tmc_obj.store_outputs()

        tmc_obj = tmc(self.Z, "treatment_tmc_outputs")
        fname = "diapause_signature.csv"
        fname = os.path.join(".", fname)
        tmc_obj.generate_matrix_from_signature_file(fname)

                    
    #====================================================
    def tmc_int(self):
        tmci_dir = ("/home/javier/Documents/"
                   "repos/too-many-cells-interactive")
        mtx_dir = ("/home/javier/Documents/persister"
                   "/data/experiment_1/with_mouse_data"
                   "/exploration/treatment_tmc_outputs"
                   "/tmci_mtx_data")

        tmc_obj = tmc(self.A, "treatment_tmc_outputs")
        tmc_obj.visualize_with_tmc_interactive(
            tmci_dir,
            "state",
            2234,
            include_matrix_data = True,
            tmci_mtx_dir=mtx_dir,
            )


obj = Explora()
# obj.load_mtx_object()
# obj.convert_mtx_object_to_h5ad()
obj.load_h5ad_object()
obj.load_metadata()
obj.replace_treatment_labels()
obj.find_mitochondrial_genes()
obj.compute_stats()
obj.plot_stats(label="_unfiltered")
obj.filter_cells_and_genes()
obj.remove_high_mitochondrial_counts()

obj.plot_stats(label="_post_mito")

obj.remove_high_total_counts_by_state()
obj.filter_cells_and_genes()

obj.compute_stats()
obj.plot_stats(label="_filtered")

obj.partition_into_states()

# obj.plot_counts_by_gene(label="_filtered")
#obj.plot_dispersion()

# obj.create_diapause_matrix()
# obj.pca_step()
# obj.diff_map_step()
# obj.create_visualization()

obj.full_tmc()
obj.tmc_int()