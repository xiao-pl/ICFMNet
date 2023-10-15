from .custom import CustomDataset


class WheatDataset(CustomDataset):

    CLASSES = ('ear', 'leaf', 'stem')

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        return instance_num, instance_pointnum, instance_cls, pt_offset_label
