import os
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

import pygame

pygame.init()
pygame.display.set_mode((400, 100))
pygame.display.set_caption("HOTAS input monitor")

devices = {}

for i in range(pygame.joystick.get_count()):
    device = pygame.joystick.Joystick(i)
    device.init()
    devices[device.get_instance_id()] = device

    print(
        f"{device.get_name()}: "
        f"{device.get_numaxes()} axes, "
        f"{device.get_numbuttons()} buttons, "
        f"{device.get_numhats()} hats"
    )

clock = pygame.time.Clock()
running = True

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            device = devices.get(getattr(event, "instance_id", None))
            if device is None:
                continue

            name = device.get_name()

            if event.type == pygame.JOYAXISMOTION:
                print(f"{name}: axis {event.axis} = {event.value:.3f}")

            elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP):
                pressed = event.type == pygame.JOYBUTTONDOWN
                print(f"{name}: button {event.button} = {pressed}")

            elif event.type == pygame.JOYHATMOTION:
                print(f"{name}: hat {event.hat} = {event.value}")

        clock.tick(100)

except KeyboardInterrupt:
    pass
finally:
    pygame.quit()