import matplotlib.pyplot as plt
import numpy as np

import matplotlib.dates as mdates


def p_cs_zoo_sidebyside(cs_dates, zoo_dates, cs_matrix, zoo_matrix):
    plt.figure(figsize=(16,12))
    plt.subplot(121)
    plt.pcolormesh(np.arange(cs_matrix.shape[1]), cs_dates, cs_matrix, shading='nearest')#,vmin=200, vmax=600)
    plt.xlabel('Transect #')
    plt.ylabel('Time')
    plt.title('a) CoastSat method')

    plt.subplot(122)
    plt.pcolormesh(np.arange(zoo_matrix.shape[1]), zoo_dates, zoo_matrix, shading='nearest')#,vmin=200, vmax=600)
    plt.xlabel('Transect #')
    plt.ylabel('Time')
    plt.title('b) Deep-learning method')
    # plt.show()

    plt.savefig("p_cs_zoo_sidebyside.png", dpi=300, bbox_inches='tight')
    plt.close()




# def p_raw_tidally_corrected_sidebyside(raw,corrected):

#     plt.figure(figsize=(12,12))
#     plt.subplot(211)
#     plt.pcolormesh( raw.index, np.arange(30,100),raw.T, shading='nearest')#,vmin=200, vmax=600)
#     plt.ylabel('Transect #')
#     plt.xlabel('Time')
#     plt.title('a) Raw tidally corrected shorelines')
#     plt.colorbar()

#     plt.subplot(212)
#     plt.pcolormesh(corrected.index,np.arange(30,100), corrected.T, shading='nearest')#,vmin=200, vmax=600)
#     plt.ylabel('Transect #')
#     plt.xlabel('Time')
#     plt.title('b) Interpolated denoised tidally corrected shorelines')
#     plt.colorbar()
#     plt.show()


# def p_shorelinechange_sidebyside(raw,corrected):

#     plt.figure(figsize=(8,8))
#     plt.pcolormesh(shore_change.index, np.arange(30,100), shore_change, vmin=0, vmax=600)
#     cb=plt.colorbar()
#     cb.set_label("Shoreline change (m)")
#     plt.ylabel('Transect #')
#     plt.xlabel('Time')
#     plt.show()


#     plt.figure(figsize=(8,8))
#     plt.contourf(df_distances_by_time_and_transect.index, np.arange(30,100), (cs_inpaint_denoised - cs_inpaint_denoised[:10,:].mean(axis=0)).T, vmin=0, vmax=600)
#     cb=plt.colorbar()
#     cb.set_label("Shoreline change (m)")
#     plt.ylabel('Transect #')
#     plt.xlabel('Time')
#     plt.show()



# plt.plot(df_distances_by_time_and_transect.index, shore_change.mean(axis=0), 'k')
# plt.ylabel('Mean shoreline displacement (m)')
# plt.show()


# plt.plot(np.arange(30,100), shore_change.mean(axis=1), 'k')
# plt.ylabel('Mean shoreline displacement (m)')
# plt.show()






# plt.figure(figsize=(16,12))

# plt.subplot(131)
# plt.pcolormesh(np.arange(zoo_matrix.shape[0]), zoo_dates, zoo_matrix.T, shading='nearest',vmin=200, vmax=600)
# plt.xlabel('Transect #')
# plt.ylabel('Time')
# plt.title('a) raw')
# plt.colorbar()

# plt.subplot(132)
# plt.pcolormesh(np.arange(zoo_matrix.shape[0]), zoo_dates, zoo_matrix_inpaint.T, shading='nearest',vmin=200, vmax=600)
# plt.xlabel('Transect #')
# plt.ylabel('Time')
# plt.title('b) inpainted')
# plt.colorbar()

# plt.subplot(133)
# plt.pcolormesh(np.arange(zoo_matrix.shape[0]), zoo_dates, kmt_inpaint_denoised.T, shading='nearest',vmin=200, vmax=600)
# plt.xlabel('Transect #')
# plt.ylabel('Time')
# plt.title('b) denoised')
# plt.colorbar()
# # plt.show()
# plt.savefig("Filtering_zoo_data.png", dpi=300, bbox_inches='tight')
# plt.close()


# plt.figure(figsize=(8,8))
# # plt.plot(zoo_dates,zoo_matrix[44,:],'k.',lw=1, label='Raw')
# plt.plot(zoo_dates,zoo_matrix_inpaint[44,:],'k',lw=1, label='Interpolated')
# plt.plot(zoo_dates,kmt_inpaint_denoised[44,:],'r',lw=2, label='Filtered')
# plt.legend()
# plt.ylabel('Shoreline location')
# # plt.show()
# plt.savefig("Filtering_zoo_data_one_transect.png", dpi=300, bbox_inches='tight')
# plt.close()



# plt.figure(figsize=(8,8))
# # plt.plot(zoo_dates,zoo_matrix[44,:],'k.',lw=1, label='Raw')
# plt.plot(cs_dates,cs_matrix[44,:],'k-o',lw=1, label='CoastSat')
# plt.plot(zoo_dates,kmt_inpaint_denoised[44,:],'r',lw=2, label='Zoo')
# plt.legend()
# plt.ylabel('Shoreline location')
# # plt.show()
# plt.savefig("cs_versus_zoo_data_one_transect.png", dpi=300, bbox_inches='tight')
# plt.close()





# plt.figure(figsize=(8,8))
# # plt.plot(zoo_dates,zoo_matrix[44,:],'k.',lw=1, label='Raw')
# plt.plot(cs_dates,cs_matrix[1,:],'k-o',lw=1, label='CS transect 2')
# plt.plot(zoo_dates,kmt_inpaint_denoised[1,:],'r',lw=2, label='Zoo transect 2')


# plt.plot(cs_dates,cs_matrix[-2,:],'m-o',lw=1, label='CS transect 85')
# plt.plot(zoo_dates,kmt_inpaint_denoised[-2,:],'c',lw=2, label='Zoo transect 85')

# plt.legend()
# plt.ylabel('Shoreline location')
# # plt.show()
# plt.savefig("cs_versus_zoo_data_two_transects.png", dpi=300, bbox_inches='tight')
# plt.close()



# plt.figure(figsize=(8,8))
# plt.contour(np.arange(zoo_matrix.shape[0]), zoo_dates, (zoo_matrix_inpaint.T - zoo_matrix_inpaint[:,0]), [0,50,100,150], colors=['k','r','g','b'])#,vmin=200, vmax=600)
# cb=plt.colorbar()
# cb.set_label("Shoreline change (m)")
# plt.xlabel('Transect #')
# plt.ylabel('Time')
# plt.show()







# # plt.figure(figsize=(12,12))

# # plt.subplot(221)
# # plt.pcolormesh(np.arange(df_distances_by_time_and_transect.shape[1]), df_distances_by_time_and_transect.index, df_distances_by_time_and_transect, shading='nearest')
# # plt.xlabel('Transect #'); plt.ylabel('Time')
# # plt.title('a) Raw tidally corrected shorelines')
# # plt.colorbar()

# # plt.subplot(222)
# # plt.pcolormesh(np.arange(df_tides_by_time_and_transect.shape[1]), df_tides_by_time_and_transect.index, df_tides_by_time_and_transect, shading='nearest')
# # plt.xlabel('Transect #'); plt.ylabel('Time')
# # plt.title('b) Tide')
# # plt.colorbar()

# # plt.subplot(223)
# # plt.pcolormesh(np.arange(df_tides_by_time_and_transect.shape[1]), df_tides_by_time_and_transect.index, df_x_by_time_and_transect, shading='nearest')
# # plt.xlabel('Transect #'); plt.ylabel('Time')
# # plt.title('c) X')
# # plt.colorbar()

# # plt.subplot(224)
# # plt.pcolormesh(np.arange(df_tides_by_time_and_transect.shape[1]), df_tides_by_time_and_transect.index, df_y_by_time_and_transect, shading='nearest')
# # plt.xlabel('Transect #'); plt.ylabel('Time')
# # plt.title('d) Y')
# # plt.colorbar()

# # plt.show()