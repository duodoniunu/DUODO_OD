from torch import nn


class DUOdoLoss(nn.Module):
    def __init__(self):
        super(DUOdoLoss, self).__init__()
        self.location_loss = nn.MSELoss()
        self.classid_loss = nn.CrossEntropyLoss()

    def forward(self,predictions,targets):
        predictions_location = predictions[ : , 0:4]
        targets_location = targets[ : , 0:4]
        predictions_classid = predictions[ : , 4:8]
        targets_classid = targets[ : , 4:8]
        predictions_loss_location = self.location_loss(predictions_location, targets_location)
        predictions_loss_classid = self.classid_loss(predictions_classid, targets_classid)
        return predictions_loss_location + predictions_loss_classid