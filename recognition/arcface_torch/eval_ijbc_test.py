'''
완성된 weight 테스트
TODO LFW, AgeDB 등 다른 validation도 함께 추가시키기
'''

from eval_ijbc_by_epoch import *
from backbones import get_model
# from utils.utils_config import get_config


# parser = argparse.ArgumentParser(
#         description="Distributed Arcface Training in Pytorch")
# parser.add_argument("config", type=str, help="py config file")
# parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
# args = parser.parse_args()

class cfg:
    weight_dir = './train_tmp/public_weight/backbone_ms1mv3_arcface_r100.pth'
    network = 'r100'
    local_rank = 0
    nproc_per_node = 4
    output = './'

backbone = get_model(
        cfg.network, dropout=0.0, fp16=True, num_features=512).cuda()
backbone.load_state_dict(torch.load(cfg.weight_dir))

# backbone = torch.nn.parallel.DistributedDataParallel(
#     module=backbone, broadcast_buffers=False, device_ids=[cfg.local_rank], bucket_cap_mb=16,
#     find_unused_parameters=True)

with torch.no_grad():
        # backbone.eval()
        ijbb_tpr_fpr_table = calculate_ijbc_score(backbone, os.path.join(cfg.output, "ijb_result"), "IJBB")
        print(ijbb_tpr_fpr_table)
        ijbc_tpr_fpr_table = calculate_ijbc_score(backbone, os.path.join(cfg.output, "ijb_result"), "IJBC")
        print(ijbc_tpr_fpr_table)
        
        
        
# +-----------+-------+-------+--------+-------+-------+-------+
# |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
# +-----------+-------+-------+--------+-------+-------+-------+
# | ijbb-IJBB | 40.24 | 92.09 | 95.47  | 96.92 | 97.84 | 98.70 |
# +-----------+-------+-------+--------+-------+-------+-------+
# +-----------+-------+-------+--------+-------+-------+-------+
# |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
# +-----------+-------+-------+--------+-------+-------+-------+
# | ijbb-IJBB | 40.24 | 92.09 | 95.47  | 96.92 | 97.84 | 98.70 |
# +-----------+-------+-------+--------+-------+-------+-------+

# +-----------+-------+-------+--------+-------+-------+-------+
# |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
# +-----------+-------+-------+--------+-------+-------+-------+
# | ijbc-IJBC | 90.99 | 95.32 | 96.81  | 97.88 | 98.54 | 99.17 |
# +-----------+-------+-------+--------+-------+-------+-------+
# +-----------+-------+-------+--------+-------+-------+-------+
# |  Methods  | 1e-06 | 1e-05 | 0.0001 | 0.001 |  0.01 |  0.1  |
# +-----------+-------+-------+--------+-------+-------+-------+
# | ijbc-IJBC | 90.99 | 95.32 | 96.81  | 97.88 | 98.54 | 99.17 |
# +-----------+-------+-------+--------+-------+-------+-------+





