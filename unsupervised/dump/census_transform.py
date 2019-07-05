import cv2

filepath = "/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/albedo/alley_1/frame_0001.png"

image = cv2.imread(filepath, 0)
height = len(image)
width = len(image[0])

#window size(3x3)
hy = 3
wx = 3
bit = 0


# for y in xrange(hy/2, height - hy/2):
#     for x in xrange(wx/2, width - wx/2):
#         census = 0
#         shift_count = 0
#         #MxN
#         for j in xrange(y - hy/2, y + hy/2 + 1):
#             for i in xrange(x - wx/2, x + wx/2 + 1):
#                 if shift_count != hy * wx / 2:
#                     census <<= 1
#                     if image[j][i] < image[y][x]:
#                         bit = 1
#                     else:
#                         bit = 0

#                     census = census + bit;
#                 shift_count += 1


#         image[y][x] = census;

# cv2.imshow("ct", image)
# cv2.imwrite("ct.png", image)
# cv2.waitKey(0)

