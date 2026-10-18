from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from inference.experimental_release import ExperimentalReleaseSelection, EXPERIMENTAL_COGNITIVE_RELEASE
from inference.tabiclv2_adapter import CognitiveReleaseContract, TabICLv2CognitiveAdapter, load_cognitive_release_contract, verify_tabiclv2_runtime
from inference.tabiclv2_context import ContextSourceAuthorization, InMemoryContextCache, load_authorized_cognitive_context
from training.contracts import FEATURE_COLUMNS, PARTICIPANT_COLUMN
from training.tabiclv2.candidate import TABICLV2_CHECKPOINT_SHA256


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class FakeModel:
    def __init__(self, probability: float = .2) -> None: self.probability = probability
    def predict_proba(self, values: object) -> np.ndarray: return np.array([[1-self.probability, self.probability]])


class Phase3AContractTests(unittest.TestCase):
    def synthetic_source(self, root: Path) -> tuple[ContextSourceAuthorization, Path]:
        source=root/'development-only.csv'; metadata=root/'provenance.json'; fields=[PARTICIPANT_COLUMN,*FEATURE_COLUMNS,'target_cognitive_concern']; rows=[]
        for index in range(500):
            rows.append({PARTICIPANT_COLUMN:f'p{index:03d}',FEATURE_COLUMNS[0]:'CP24M',**{field:'1' for field in FEATURE_COLUMNS[1:]},'target_cognitive_concern':'1' if index<53 else '0'})
        with source.open('w',newline='',encoding='utf-8') as handle: writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        membership=hashlib.sha256('\n'.join(sorted(row[PARTICIPANT_COLUMN] for row in rows)).encode()).hexdigest().upper(); source_hash=digest(source)
        metadata.write_text(json.dumps({'schema':'tabiclv2-cognitive-development-context-v1','source_sha256':source_hash,'membership_sha256':membership,'dataset_sha256':'dataset','split_fingerprint_sha256':'split','participants':500,'cognitive_positive_labels':53}))
        return ContextSourceAuthorization(source,metadata,source_hash,membership,'dataset','split',root/'combined-625.csv'),source
    def test_authorized_source_and_cache(self):
        with tempfile.TemporaryDirectory() as value:
            authorization,_=self.synthetic_source(Path(value)); context=load_authorized_cognitive_context(authorization);self.assertEqual((500,16),context.feature_rows.shape);self.assertEqual(53,int(context.cognitive_labels.sum()));self.assertFalse(context.feature_rows.flags.writeable);self.assertNotIn('p000',repr(context))
            calls=[];cache=InMemoryContextCache(lambda: calls.append(1) or FakeModel());self.assertIs(cache.get(),cache.get());self.assertEqual([1],calls)
    def test_context_rejects_combined_changed_or_missing_source(self):
        with tempfile.TemporaryDirectory() as value:
            authorization,source=self.synthetic_source(Path(value))
            combined=ContextSourceAuthorization(source,authorization.provenance_path,authorization.approved_source_sha256,authorization.approved_membership_sha256,'dataset','split',source)
            with self.assertRaises(ValueError):load_authorized_cognitive_context(combined)
            source.write_text('changed')
            with self.assertRaises(ValueError):load_authorized_cognitive_context(authorization)
            with self.assertRaises(FileNotFoundError):load_authorized_cognitive_context(ContextSourceAuthorization(Path(value)/'none.csv',authorization.provenance_path,'x','x','dataset','split',Path(value)/'combined.csv'))
    def test_context_rejects_schema_and_label_contract_changes(self):
        with tempfile.TemporaryDirectory() as value:
            root=Path(value); authorization,source=self.synthetic_source(root); metadata=json.loads(authorization.provenance_path.read_text())
            source.write_text('participant_key,wrong\np001,x\n'); source_hash=digest(source); metadata['source_sha256']=source_hash;authorization.provenance_path.write_text(json.dumps(metadata)); altered=ContextSourceAuthorization(source,authorization.provenance_path,source_hash,authorization.approved_membership_sha256,'dataset','split',root/'combined.csv')
            with self.assertRaises(ValueError):load_authorized_cognitive_context(altered)
    def test_adapter_calibration_threshold_and_invalid_probability(self):
        release=CognitiveReleaseContract(TABICLV2_CHECKPOINT_SHA256,1.069105693175016,.255897163477147,.1503614235173848);adapter=TabICLv2CognitiveAdapter(InMemoryContextCache(lambda:FakeModel(.2)),release);values=('CP24M',*('1',)*15);self.assertTrue(0<adapter.positive_probability(values)<1);self.assertEqual(adapter.positive_probability(values)>=release.threshold,adapter.decision(values))
        with self.assertRaises(ValueError):adapter.positive_probability(values[:-1])
        with self.assertRaises(RuntimeError):TabICLv2CognitiveAdapter(InMemoryContextCache(lambda:FakeModel(float('nan'))),release).positive_probability(values)
    def test_manifest_runtime_and_explicit_selection_contracts(self):
        with tempfile.TemporaryDirectory() as value:
            root=Path(value);manifest=root/'manifest.json';manifest.write_text(json.dumps({'status':'development_only_experimental_not_deployed','domain':'cognitive','checkpoint_sha256':TABICLV2_CHECKPOINT_SHA256,'feature_contract':list(FEATURE_COLUMNS),'calibration':{'coefficient':1.069105693175016,'intercept':.255897163477147},'threshold':{'value':.1503614235173848}}));self.assertAlmostEqual(.1503614235173848,load_cognitive_release_contract(manifest).threshold)
            checkpoint=root/'tabicl-classifier-v2-20260212.ckpt';checkpoint.write_bytes(b'x')
            with patch('inference.tabiclv2_adapter._sha256',return_value=TABICLV2_CHECKPOINT_SHA256),patch('inference.tabiclv2_adapter.importlib.metadata.version',return_value='2.2.0'):verify_tabiclv2_runtime(checkpoint)
            with patch('inference.tabiclv2_adapter._sha256',return_value='bad'):
                with self.assertRaises(ValueError):verify_tabiclv2_runtime(checkpoint)
            with patch('inference.tabiclv2_adapter._sha256',return_value=TABICLV2_CHECKPOINT_SHA256),patch('inference.tabiclv2_adapter.importlib.metadata.version',return_value='wrong'):
                with self.assertRaises(ValueError):verify_tabiclv2_runtime(checkpoint)
            ExperimentalReleaseSelection(EXPERIMENTAL_COGNITIVE_RELEASE).require_experimental_cognitive()
            with self.assertRaises(ValueError):ExperimentalReleaseSelection('historical').require_experimental_cognitive()
