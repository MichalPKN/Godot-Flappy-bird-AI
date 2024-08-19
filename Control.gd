extends Control


# Declare member variables here. Examples:
# var a = 2
# var b = "text"


# Called when the node enters the scene tree for the first time.
func _ready():
	_on_CheckButton_toggled(false)


# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass


func _on_CheckButton_toggled(button_pressed):
	if button_pressed:
		Engine.time_scale = 1
		Engine.target_fps = 60
	else:
		Engine.time_scale = 3
		Engine.target_fps = 90
