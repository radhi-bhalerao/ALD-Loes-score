import importlib

roi_dict = {
    'tissue_seg': {'sub_dir': None, 'measure': 'volume', 'type': 'derived'},
    'VentriclesDist': {'sub_dir': None, 'measure': 'distance', 'type': 'derived'},
    'template': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'template',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'nihpd', 'oasis_to_nihpd_BrainCerebellum.nii.gz'
        ),
    },
    'template_mask': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'aux',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'nihpd', 'oasis_to_nihpd_BrainCerebellumMask.nii.gz'
        ),
        'regions': {
            'Brain': [1],
        },
    },
    
    'lobes': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'template',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'nihpd', 'oasis_to_nihpd_BrainLobesfinal.nii.gz'
        ),
        'regions': {
            'Frontal': [1],
            'Parietal': [2],
            'Occipital': [3],
            'Temporal': [4, 5, 6],
            'AnteriorTemporal': [4],
            'MiddleTemporal': [5],
            'PosteriorTemporal': [6],
            'Parietal_Occipital': [2, 3],
        },
    }, 
    
    'CorpusCallosum': {
        'sub_dir': 'TemplateReg',
        'measure': 'volume',
        'type': 'template',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'nihpd', 'oasis_to_nihpd_CorpusCallosum.nii.gz'
        ),
        'regions': {
            'CorpusCallosum': [1, 2, 3, 4, 5],
            'CorpusCallosum_Rostrum': [1],
            'CorpusCallosum_Genu': [2],
            'CorpusCallosum_Body': [3],
            'CorpusCallosum_Isthmus': [4],
            'CorpusCallosum_Splenium': [5]
        }
    },

    'ald_rois' : {
        'sub_dir' : 'TemplateReg',
        'measure': 'volume',
        'type' : 'template',
        'source': importlib.resources.files('pyalfe').joinpath(
            'templates', 'nihpd', 'oasis_to_nihpd_ALD_ROIs.nii.gz'),
        'regions' : {
            "Right LGN": [1],
            "Left LGN": [2],
            "Right OR": [3],
            "Left OR": [4],
            "Right Optic Tract": [5],
            "Left Optic Tract": [6],
            "Right Meyers Loop": [7],
            "Left Meyers Loop": [8],
            "Right Medial Geniculate": [9],
            "Left Medial Geniculate": [10],
            "Right Brachium Inf Col": [11],
            "Left Brachium Inf Col": [12],
            "Right Lat Lemniscus": [13],
            "Left Lat Lemniscus": [14],
            "Pons": [15],
            "Right Anterior Thalamus":[16],
            "Left Anterior Thalamus": [17],
            "Right ALIC": [18],
            "Left ALIC": [19],
            "Right CP Frontopontine": [20],
            "Left CP Frontopontine": [21]
        }
      },

    'basal_ganglia' : {
    'sub_dir' : 'TemplateReg',
    'measure': 'volume',
    'type' : 'template',
    'source': importlib.resources.files('pyalfe').joinpath(
        'templates', 'nihpd', 'oasis_to_nihpd_BasalGangliafinal.nii.gz'),
    'regions' : {
        "Basal Ganglia": [1]
    }
    }

}
