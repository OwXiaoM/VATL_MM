import unittest
import numpy as np
from vesseltok.break_helpers import arc_weights,true_runs,pair_transitions,neighbor_branch_distance,weighted_summary,exposure_bins,require_audit_scope


class BreakAuditTests(unittest.TestCase):
    def test_arc_weights_duplicate_endpoints_do_not_add_length(self):
        arc=np.array([0,.25,.5,.5,.75,1.]);w=arc_weights(arc)
        self.assertAlmostEqual(w.sum(),1.);self.assertAlmostEqual(w[2]+w[3],.25)

    def test_partition_transitions_match_bruteforce_with_missing_foreground(self):
        rng=np.random.default_rng(42)
        for _ in range(30):
            a,b,t=rng.integers(0,4,(3,12));r=pair_transitions(a,b,t);counts={k:0 for k in ['removed_excess_pairs','gained_excess_pairs','removed_target_connections','gained_target_connections']}
            for i in range(12):
                for j in range(i):
                    ca=a[i]>0 and a[i]==a[j];cb=b[i]>0 and b[i]==b[j];ct=t[i]>0 and t[i]==t[j]
                    if ca and not cb:counts['removed_target_connections' if ct else 'removed_excess_pairs']+=1
                    if cb and not ca:counts['gained_target_connections' if ct else 'gained_excess_pairs']+=1
            self.assertEqual({k:r[k] for k in counts},counts)

    def test_nonincident_branch_context_excludes_shared_junction(self):
        p=np.array([[0,0,0],[0,1,0],[4,0,0],[4,1,0],[0,2,0]],float);ids=np.array([0,0,1,1,2]);paths=[[0,1],[2,3],[1,4]]
        d=neighbor_branch_distance(p,ids,np.ones(5,bool),paths)
        self.assertAlmostEqual(d[0],4.);self.assertGreater(d[-1],4.)
        self.assertTrue(np.isinf(neighbor_branch_distance(p,ids,np.ones(5,bool),paths,radius=2.)).all())

    def test_scope_rejects_evaluation_and_duplicate_people(self):
        splits={'train':['IXI001-A'],'validation':['IXI002-B'],'evaluation':['IXI003-C']}
        require_audit_scope(['IXI001-A','IXI002-B'],splits)
        with self.assertRaises(ValueError):require_audit_scope(['IXI003-C'],splits)
        with self.assertRaises(ValueError):require_audit_scope(['IXI001-A','IXI001-other'],splits)

    def test_exposure_denominator_uses_baseline_supported_arc(self):
        w=np.array([1.,2.,1.]);a=np.array([.8,.7,.2]);b=np.array([.4,.6,.1]);r=weighted_summary(w,a,b,a>=.5)
        self.assertEqual(r['baseline_supported_arc_mm'],3.);self.assertAlmostEqual(r['new_loss_arc_fraction'],1/3)
        rows=exposure_bins(np.array([1.,np.inf,np.nan]),[0,2,np.inf],w,a,b,a>=.5)
        self.assertEqual(rows[-1]['baseline_supported_arc_mm'],2.)

    def test_runs_are_complete_with_boundary_gaps(self):
        self.assertEqual(true_runs([1,1,0,1,0,1]),[(0,2),(3,4),(5,6)])


if __name__=='__main__':unittest.main()
