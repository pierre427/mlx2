"""Real temporary Git repositories; MLX/native imports prohibited."""
import importlib.abc
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name=='mlx' or name.startswith('mlx.') or name=='_paged_kv_native':raise RuntimeError('runtime forbidden')
sys.meta_path.insert(0,Guard());sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mlx2.runtime import paged_price_identity as I

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.root=self.base/'repo';self.root.mkdir()
        home=self.base/'home';home.mkdir();self.env=patch.dict(os.environ,{'HOME':str(home),'XDG_CONFIG_HOME':str(home/'.config')});self.env.start()
        self.root_patch=patch.object(I,'ROOT',self.root);self.root_patch.start();I._SOURCE_MEMO.clear();I._IDENTITY_MEMO.clear()
        self.git('init','-q');self.git('config','user.name','CPU');self.git('config','user.email','cpu@example.invalid')
        (self.root/'code.py').write_text('initial\n');(self.root/'nested').mkdir();(self.root/'nested/other.py').write_text('stable\n')
        (self.root/'.gitignore').write_text('.cache/\n');self.git('add','.');self.git('commit','-qm','initial');self.commit=self.git('rev-parse','HEAD').strip()
    def tearDown(self):self.root_patch.stop();self.env.stop();self.temp.cleanup();I._SOURCE_MEMO.clear();I._IDENTITY_MEMO.clear()
    def git(self,*args):return subprocess.check_output(['git',*args],cwd=self.root,text=True,stderr=subprocess.DEVNULL)
    def seed(self):self.assertTrue(I._source_unchanged(self.commit))
    def test_stable_inventory_needs_no_subprocess_and_canonical_alias_reuses(self):
        self.seed();alias=self.base/'alias';alias.symlink_to(self.root,target_is_directory=True)
        with patch.object(I.subprocess,'check_output',side_effect=AssertionError('unexpected Git process')):
            self.assertTrue(I._source_unchanged(self.commit))
            with patch.object(I,'ROOT',alias):self.assertTrue(I._source_unchanged(self.commit))
    def test_content_with_restored_mtime_poisoned_even_after_bytes_restored(self):
        self.seed();p=self.root/'code.py';st=p.stat();p.write_text('changed\n');os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))
        self.assertFalse(I._source_unchanged(self.commit));p.write_text('initial\n');self.assertFalse(I._source_unchanged(self.commit))
    def test_mtime_only_change_refuses(self):
        self.seed();p=self.root/'code.py';st=p.stat();os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns+10_000_000))
        self.assertFalse(I._source_unchanged(self.commit))
    def test_inode_replace_refuses(self):
        self.seed();p=self.root/'code.py';replacement=self.base/'replacement';replacement.write_bytes(p.read_bytes());os.replace(replacement,p)
        self.assertFalse(I._source_unchanged(self.commit))
    def test_mode_change_refuses_even_if_git_ignores_filemode(self):
        self.git('config','core.filemode','false');self.seed();p=self.root/'code.py';p.chmod(p.stat().st_mode ^ 0o100)
        self.assertFalse(I._source_unchanged(self.commit))
    def test_new_untracked_nested_file_or_directory_requires_git_and_refuses(self):
        self.seed();(self.root/'nested/new.py').write_text('new')
        with patch.object(I.subprocess,'check_output',wraps=subprocess.check_output) as calls:
            self.assertFalse(I._source_unchanged(self.commit));self.assertTrue(any('status' in c.args[0] for c in calls.call_args_list))
    def test_ignored_directory_change_checks_git_then_refreshes(self):
        self.seed();cache=self.root/'.cache';cache.mkdir();(cache/'temp').write_text('ignored')
        self.assertTrue(I._source_unchanged(self.commit))
        with patch.object(I.subprocess,'check_output',side_effect=AssertionError('Git on stable refreshed inventory')):
            self.assertTrue(I._source_unchanged(self.commit))
    def test_head_ref_change_refuses(self):
        self.seed();self.git('commit','--allow-empty','-qm','new HEAD');self.assertFalse(I._source_unchanged(self.commit))
    def test_unrelated_worktree_branch_and_commit_do_not_refresh_this_checkout(self):
        self.seed();self.git('branch','unrelated')
        other=self.base/'other-worktree'
        subprocess.run(['git','worktree','add','-q',str(other),'unrelated'],cwd=self.root,check=True)
        subprocess.run(['git','commit','--allow-empty','-qm','other branch'],cwd=other,check=True)
        with patch.object(I.subprocess,'check_output',side_effect=AssertionError('unrelated ref caused Git status')):
            self.assertTrue(I._source_unchanged(self.commit))
    def test_current_symbolic_chain_retarget_checks_actual_git_cleanliness(self):
        self.git('branch','first');self.git('branch','second');self.seed()
        self.git('symbolic-ref','refs/heads/proxy','refs/heads/first')
        self.git('symbolic-ref','HEAD','refs/heads/proxy')
        with patch.object(I.subprocess,'check_output',wraps=subprocess.check_output) as calls:
            self.assertTrue(I._source_unchanged(self.commit))
            self.assertTrue(any('status' in c.args[0] for c in calls.call_args_list))
        self.git('symbolic-ref','refs/heads/proxy','refs/heads/second')
        with patch.object(I.subprocess,'check_output',wraps=subprocess.check_output) as calls:
            self.assertTrue(I._source_unchanged(self.commit))
            self.assertTrue(any('status' in c.args[0] for c in calls.call_args_list))
    def test_current_packed_ref_change_checks_actual_git_cleanliness(self):
        self.seed();self.git('pack-refs','--all','--prune')
        with patch.object(I.subprocess,'check_output',wraps=subprocess.check_output) as calls:
            self.assertTrue(I._source_unchanged(self.commit))
            self.assertTrue(any('status' in c.args[0] for c in calls.call_args_list))
    def test_staged_index_change_refuses_even_with_same_working_bytes(self):
        self.seed();self.git('rm','--cached','code.py');self.assertFalse(I._source_unchanged(self.commit))
    def test_index_refresh_requires_actual_git_cleanliness(self):
        self.seed();self.git('update-index','--assume-unchanged','code.py')
        with patch.object(I.subprocess,'check_output',wraps=subprocess.check_output) as calls:
            self.assertTrue(I._source_unchanged(self.commit));self.assertTrue(any('status' in c.args[0] for c in calls.call_args_list))
        (self.root/'code.py').write_text('hidden from git\n');self.assertFalse(I._source_unchanged(self.commit))
    def test_symlink_target_bytes_and_alias_retarget_refuse(self):
        target=self.base/'target';target.write_text('bound');link=self.root/'link.py';link.symlink_to(target)
        self.git('add','link.py');self.git('commit','-qm','link');self.commit=self.git('rev-parse','HEAD').strip();self.seed()
        target.write_text('drift');self.assertFalse(I._source_unchanged(self.commit))
    def test_tracked_symlink_directory_refuses_unbound_descendants(self):
        directory=self.base/'outside';directory.mkdir();(directory/'code.py').write_text('unbound')
        (self.root/'linked-directory').symlink_to(directory,target_is_directory=True)
        self.git('add','linked-directory');self.git('commit','-qm','directory link');self.commit=self.git('rev-parse','HEAD').strip()
        self.assertFalse(I._source_unchanged(self.commit))
    def test_symlink_alias_retarget_with_same_bytes_refuses(self):
        first=self.base/'first';second=self.base/'second';first.write_text('same');second.write_text('same')
        link=self.root/'link.py';link.symlink_to(first);self.git('add','link.py');self.git('commit','-qm','link');self.commit=self.git('rev-parse','HEAD').strip();self.seed()
        link.unlink();link.symlink_to(second);self.assertFalse(I._source_unchanged(self.commit))

    def test_global_ignore_change_detects_existing_untracked_name(self):
        excludes=self.base/'ignore';excludes.write_text('untracked.txt\n');self.git('config','--global','core.excludesFile',str(excludes))
        (self.root/'untracked.txt').write_text('initially ignored');self.seed();excludes.write_text('nothing\n')
        self.assertFalse(I._source_unchanged(self.commit))
    def test_untracked_ignored_gitignore_content_change_checks_cleanliness(self):
        # A local ignore file can be ignored itself yet still affect Git status.
        excludes=self.base/'ignore';excludes.write_text('.gitignore\nuntracked.txt\n')
        self.git('config','--global','core.excludesFile',str(excludes))
        (self.root/'nested/.gitignore').write_text('new.py\n');(self.root/'nested/new.py').write_text('initially ignored')
        self.seed();(self.root/'nested/.gitignore').write_text('other.py\n')
        self.assertFalse(I._source_unchanged(self.commit))
    def test_predeclared_missing_git_config_include_creation_is_detected(self):
        included=self.base/'included-config';self.git('config','--global','include.path',str(included));self.seed()
        included.write_text('[core]\n excludesFile = '+str(self.base/'ignore')+'\n')
        with patch.object(I.subprocess,'check_output',wraps=subprocess.check_output) as calls:
            self.assertTrue(I._source_unchanged(self.commit));self.assertTrue(any('status' in c.args[0] for c in calls.call_args_list))

    def test_git_environment_drift_refuses(self):
        self.seed()
        with patch.dict(os.environ,{'GIT_INDEX_FILE':str(self.base/'other-index')}):self.assertFalse(I._source_unchanged(self.commit))
    def test_refresh_raced_untracked_name_is_not_absorbed_into_clean_memo(self):
        self.seed();(self.root/'.cache').mkdir() # ignored change starts refresh.
        original=I._source_directories
        def raced(root):
            (self.root/'untracked-after-status.py').write_text('race');return original(root)
        with patch.object(I,'_source_directories',side_effect=raced):self.assertFalse(I._source_unchanged(self.commit))
    def test_metadata_race_after_final_head_check_is_not_memoized(self):
        original=I._git_source_clean;calls=[]
        def raced(root,commit):
            clean=original(root,commit);calls.append(1)
            if len(calls)==2:self.git('commit','--allow-empty','-qm','raced HEAD')
            return clean
        with patch.object(I,'_git_source_clean',side_effect=raced):self.assertFalse(I._source_unchanged(self.commit))

    def test_initial_full_hash_source_race_never_installs_identity(self):
        files=tuple(self.base/name for name in ('manifest','wheel','native'))
        for p in files:p.write_text('bound')
        def fake_compute(*a,**kw):
            (self.root/'code.py').write_text('raced\n');return {'source_commit':self.commit}
        with patch.object(I,'_watched_paths',return_value=files),patch.object(I,'compute_live_price_identity',side_effect=fake_compute):
            with self.assertRaisesRegex(RuntimeError,'during attestation'):
                I.cached_live_price_identity(*files,adapter_artifact_root=self.root)
        self.assertEqual(I._IDENTITY_MEMO,{})

if __name__=='__main__':unittest.main()
